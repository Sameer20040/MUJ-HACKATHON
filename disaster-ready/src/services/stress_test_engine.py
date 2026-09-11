"""
Stress Test Engine — agent-simulates a Scenario against a ResponsePlan.

Produces honest, actionable output: per-timestep evacuation success, resource coverage,
detected bottlenecks, cascade failures, and CONCRETE revision recommendations.
"""
import json
import math
from datetime import datetime
from typing import Dict, List, Optional

from shapely import wkt as shapely_wkt
from shapely.geometry import Point
from sqlalchemy.orm import Session

from src.models import (
    Scenario, ScenarioEvent, ResponsePlan, ResourcePosition, EvacuationRoute, PlanStressTest,
)


def _distance_km(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    dy = (lat2 - lat1) * 111.0
    return math.hypot(dx, dy)


class HazardAgent:
    """Drives hazard evolution over the scenario horizon."""
    def __init__(self, scenario: Scenario, events: List[ScenarioEvent]):
        self.scenario = scenario
        self.events = sorted(events, key=lambda e: e.timestamp_offset_hours)

    def timeline(self):
        """Yield (offset_hours, event) in order."""
        for ev in self.events:
            yield ev.timestamp_offset_hours, ev


class ExposureAgent:
    """Estimates population exposure at a hazard epicenter given severity."""
    POP_PER_KM2_BASE = 800.0
    SEV_MULTIPLIER = {"low": 0.2, "medium": 0.6, "high": 1.0, "extreme": 1.6}

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    def population_at_risk(self) -> float:
        poly = shapely_wkt.loads(self.scenario.affected_area)
        area_km2 = poly.area * (111.0 ** 2)  # rough degrees->km2 for small polygons
        mult = self.SEV_MULTIPLIER.get(
            self.scenario.severity.value if hasattr(self.scenario.severity, "value") else str(self.scenario.severity), 1.0
        )
        return self.POP_PER_KM2_BASE * max(area_km2, 1.0) * mult


class ResourceAgent:
    """Checks each resource position against scenario epicenter."""
    def __init__(self, resources: List[ResourcePosition]):
        self.resources = resources

    def coverage(self, epicenter: Point, horizon_h: float) -> Dict:
        covered = 0
        total = max(len(self.resources), 1)
        gaps = []
        for r in self.resources:
            pt = shapely_wkt.loads(r.location)
            d_km = _distance_km(epicenter.x, epicenter.y, pt.x, pt.y)
            # road speed ~45km/h + prep 0.5h
            eta = 0.5 + d_km / 45.0
            deploy = r.deployment_time_hours or 0.0
            usable = eta + deploy <= horizon_h
            if usable:
                covered += 1
            else:
                gaps.append({
                    "resource_id": r.id,
                    "resource_type": r.resource_type,
                    "quantity": r.quantity,
                    "distance_km": round(d_km, 1),
                    "eta_hours": round(eta, 2),
                    "deployment_time_hours": deploy,
                    "horizon_hours": horizon_h,
                    "issue": "Cannot arrive+deploy within scenario horizon",
                })
        return {"coverage": covered / total, "gaps": gaps}


class ResponseAgent:
    """Checks evacuation route capacity against population demand."""
    def __init__(self, routes: List[EvacuationRoute]):
        self.routes = routes

    def evacuation(self, population: float, epicenter: Point, horizon_h: float, severity: str) -> Dict:
        total_capacity = 0.0
        bottlenecks = []
        usable_routes = 0
        for route in self.routes:
            path = shapely_wkt.loads(route.path)
            if path.geom_type == "LineString":
                xs, ys = path.xy
                d_km = _distance_km(epicenter.x, epicenter.y, xs[0], ys[0])
            else:
                d_km = 50.0
            cap = (route.capacity_per_hour or 0) * horizon_h * 0.6  # 60% utilization realistic
            # hazard severity reduces route throughput
            sev_mult = {"low": 1.0, "medium": 0.85, "high": 0.65, "extreme": 0.45}.get(severity, 1.0)
            cap *= sev_mult
            if cap <= 0:
                bottlenecks.append({
                    "route_id": route.id, "route_name": route.route_name,
                    "issue": "Zero capacity in plan", "suggestion": "Define capacity_per_hour for this route",
                })
                continue
            usable_routes += 1
            total_capacity += cap
            if d_km > 30:
                bottlenecks.append({
                    "route_id": route.id, "route_name": route.route_name,
                    "issue": f"Route origin {round(d_km,1)} km from epicenter — starts too far away",
                    "suggestion": f"Pre-stage assembly point within 30 km of high-risk zone",
                })
            vulnerable = json.loads(route.vulnerable_segments) if route.vulnerable_segments else []
            if vulnerable:
                bottlenecks.append({
                    "route_id": route.id, "route_name": route.route_name,
                    "issue": f"{len(vulnerable)} vulnerable segment(s) (e.g. {vulnerable[0].get('reason','unknown')})",
                    "suggestion": "Mark alternate route for these segments in plan revision",
                })

        success_rate = min(1.0, total_capacity / max(population, 1.0)) if population > 0 else 1.0
        if usable_routes == 0:
            bottlenecks.append({
                "route_id": None, "route_name": None,
                "issue": "No usable evacuation routes for this hazard",
                "suggestion": "Add at least 2 routes with measured capacity",
            })
        return {"success_rate": round(success_rate, 3), "capacity": total_capacity,
                "population": round(population), "bottlenecks": bottlenecks}


class CascadeAgent:
    """Detects cascade events that the plan does not cover."""
    COVERED_TYPES = {"flood", "landslide", "power_grid", "fire", "water_contamination", "bridge_failure", "road_blockage", "tree_fall"}

    def failures(self, events: List[ScenarioEvent], plan: ResponsePlan) -> List[Dict]:
        detected = []
        plan_data = {}
        if plan.plan_data:
            try:
                plan_data = json.loads(plan.plan_data)
            except Exception:
                plan_data = {}
        covered = set(plan_data.get("covered_cascades", [])) | self.COVERED_TYPES & {plan.hazard_type.value}
        for ev in events:
            if ev.event_type == "cascade_step":
                etype = (ev.description or "").split()[0].lower()
                if ev.description and not any(c in (ev.description or "").lower() for c in covered):
                    detected.append({
                        "offset_hours": ev.timestamp_offset_hours,
                        "description": ev.description,
                        "severity_impact": ev.magnitude,
                        "note": "Cascade event not covered by current plan triggers",
                    })
        return detected


class StressTestEngine:
    """Runs the full multi-agent simulation of plan vs scenario."""

    def run(self, db: Session, plan: ResponsePlan, scenario: Scenario) -> PlanStressTest:
        events = db.query(ScenarioEvent).filter(ScenarioEvent.scenario_id == scenario.id).all()
        resources = db.query(ResourcePosition).filter(ResourcePosition.plan_id == plan.id).all()
        routes = db.query(EvacuationRoute).filter(EvacuationRoute.plan_id == plan.id).all()

        hazard = HazardAgent(scenario, events)
        exposure = ExposureAgent(scenario)
        resource_agent = ResourceAgent(resources)
        response_agent = ResponseAgent(routes)
        cascade_agent = CascadeAgent()

        # Epicenter = first event location
        epicenter = Point(0, 0)
        if events:
            epicenter = shapely_wkt.loads(events[0].location)
        horizon = float(scenario.time_horizon_hours or 24)

        population = exposure.population_at_risk()
        evac = response_agent.evacuation(population, epicenter, horizon,
                                         scenario.severity.value if hasattr(scenario.severity, "value") else str(scenario.severity))
        res = resource_agent.coverage(epicenter, horizon)
        cascades = cascade_agent.failures(events, plan)

        time_to_response = 24.0  # worst case default
        if evac["success_rate"] >= 0.9 and res["coverage"] >= 0.8:
            time_to_response = horizon * 0.3
        elif evac["success_rate"] >= 0.6:
            time_to_response = horizon * 0.6
        else:
            time_to_response = horizon
        time_to_response = round(time_to_response, 1)

        overall = round(100 * (0.45 * evac["success_rate"] + 0.35 * res["coverage"] + 0.20 * max(0.0, 1 - time_to_response / horizon)), 1)

        # ---- Concrete revisions from detected bottlenecks ----
        revisions = {"routes": [], "resources": [], "thresholds": []}
        for b in evac["bottlenecks"]:
            if b.get("route_id") and "capacity" not in b["issue"].lower():
                revisions["routes"].append({"action": "review_route", "route_id": b["route_id"], "reason": b["issue"], "suggestion": b["suggestion"]})
            elif b.get("route_id"):
                revisions["routes"].append({"action": "set_capacity", "route_id": b["route_id"], "suggestion": b["suggestion"]})
            else:
                revisions["routes"].append({"action": "add_routes", "suggestion": b["suggestion"]})
        for gap in res["gaps"]:
            res_row = db.query(ResourcePosition).get(gap["resource_id"])
            pt = shapely_wkt.loads(res_row.location) if res_row else Point(0, 0)
            ex, ey = epicenter.x, epicenter.y
            if res_row and gap["distance_km"] <= 150:
                # Local depot that's too slow — recommend relocating halfway toward epicenter
                mid_lon = (pt.x + ex) / 2
                mid_lat = (pt.y + ey) / 2
                revisions["resources"].append({
                    "action": "relocate",
                    "resource_id": gap["resource_id"],
                    "from": f"POINT({pt.x:.5f} {pt.y:.5f})",
                    "to": f"POINT({mid_lon:.5f} {mid_lat:.5f})",
                    "reason": f"{gap['resource_type']} {gap['distance_km']} km away cannot deploy within {horizon}h horizon",
                    "benefit": "Roughly halves deployment time",
                })
            else:
                # Cross-region depot — moving it would abandon its home region; pre-position instead
                poly = shapely_wkt.loads(scenario.affected_area)
                centroid = poly.centroid
                revisions["resources"].append({
                    "action": "preposition_new",
                    "resource_type": gap["resource_type"],
                    "suggested_location": f"POINT({centroid.x:.5f} {centroid.y:.5f})",
                    "suggested_quantity": max(gap.get("quantity", 100) // 2, 10),
                    "reason": f"Nearest {gap['resource_type']} depot is {gap['distance_km']} km away (cross-region) — pre-position a forward cache inside the affected area",
                })
        for c in cascades:
            revisions["thresholds"].append({
                "action": "add_cascade_trigger",
                "trigger": c["description"],
                "at_offset_hours": c["offset_hours"],
            })

        stress = PlanStressTest(
            plan_id=plan.id,
            scenario_id=scenario.id,
            evacuations_success_rate=evac["success_rate"],
            resource_coverage=res["coverage"],
            time_to_adequate_response_hours=time_to_response,
            bottlenecks=json.dumps(evac["bottlenecks"] + res["gaps"]),
            cascading_failures_detected=json.dumps(cascades),
            overall_score=overall,
            recommended_revision=json.dumps(revisions),
        )
        db.add(stress)
        plan.stress_test_score = overall
        db.commit()
        db.refresh(stress)
        return stress

    def _res_location(self, db: Session, resource_id) -> str:
        r = db.query(ResourcePosition).get(resource_id)
        return r.location if r else "POINT(0 0)"