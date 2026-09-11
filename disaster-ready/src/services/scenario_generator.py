"""
Scenario Generator — derives plausible disaster scenarios from REAL live sensor readings.

No fabricated data: every scenario is anchored to a real SensorReading from the DB.
Trend analysis + threshold proximity drive the probability and severity.
"""
import math
import random
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
from shapely import wkt as shapely_wkt
from shapely.geometry import Point
from sqlalchemy.orm import Session

from src.models import (
    HazardType, Severity, ScenarioStatus,
    SensorReading, SensorSite, Scenario, ScenarioEvent,
)

# ---------------------------------------------------------------------------
# Thresholds per hazard/reading type (configurable — these are trigger levels)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "earthquake": {"units": "magnitude", "levels": [(4.0, "low"), (5.0, "medium"), (6.0, "high"), (7.0, "extreme")]},
    "rainfall":   {"units": "mm",       "levels": [(5, "low"), (20, "medium"), (50, "high"), (100, "extreme")]},
    "flood":      {"units": "ft",       "levels": [(4, "low"), (8, "medium"), (12, "high"), (18, "extreme")]},
    "wind_speed": {"units": "km/h",     "levels": [(60, "low"), (90, "medium"), (120, "high"), (160, "extreme")]},
    "wind_gust":  {"units": "km/h",     "levels": [(80, "low"), (110, "medium"), (140, "high"), (180, "extreme")]},
    "temperature":{"units": "°C",       "levels": [(35, "low"), (40, "medium"), (45, "high"), (50, "extreme")]},
    "humidity":   {"units": "%",        "levels": [(85, "low"), (90, "medium"), (95, "high"), (99, "extreme")]},
}

# Cascade rules from the build spec
CASCADE_RULES = {
    "earthquake": [
        {"type": "landslide", "min_magnitude": 4.5, "description": "Slope failure triggered by shaking", "offset_h": 0.5, "prob": 0.35},
        {"type": "tsunami", "min_magnitude": 6.0, "description": "Coastal tsunami wave generated", "offset_h": 1.0, "prob": 0.15},
        {"type": "dam_failure", "min_magnitude": 5.0, "description": "Dam structural compromise", "offset_h": 2.0, "prob": 0.10},
        {"type": "power_grid", "min_magnitude": 4.0, "description": "Power grid substation failure", "offset_h": 0.25, "prob": 0.45},
        {"type": "fire", "min_magnitude": 5.5, "description": "Gas line rupture ignition", "offset_h": 1.5, "prob": 0.12},
    ],
    "flood": [
        {"type": "landslide", "min_value": 8.0, "description": "Saturated slope failure", "offset_h": 3.0, "prob": 0.25},
        {"type": "water_contamination", "min_value": 6.0, "description": "Sewage treatment overflow", "offset_h": 6.0, "prob": 0.40},
        {"type": "bridge_failure", "min_value": 10.0, "description": "Bridge scour undermining", "offset_h": 4.0, "prob": 0.15},
        {"type": "power_grid", "min_value": 9.0, "description": "Substation inundation", "offset_h": 5.0, "prob": 0.30},
    ],
    "rainfall": [
        {"type": "flood", "min_value": 20.0, "description": "Flash flooding from sustained rainfall", "offset_h": 2.0, "prob": 0.45},
        {"type": "landslide", "min_value": 35.0, "description": "Saturated slope failure", "offset_h": 6.0, "prob": 0.20},
        {"type": "road_blockage", "min_value": 25.0, "description": "Road washout / debris", "offset_h": 4.0, "prob": 0.35},
    ],
    "wind_speed": [
        {"type": "power_grid", "min_value": 90.0, "description": "Wind damage to transmission lines", "offset_h": 1.0, "prob": 0.40},
        {"type": "structural_damage", "min_value": 120.0, "description": "Roof / structure damage", "offset_h": 2.0, "prob": 0.30},
     ],
    "wind_gust": [
        {"type": "power_grid", "min_value": 110.0, "description": "Gust damage to power lines", "offset_h": 0.5, "prob": 0.45},
        {"type": "tree_fall", "min_value": 130.0, "description": "Tree fall blocking roads", "offset_h": 1.0, "prob": 0.50},
    ],
}


def _severity_from_value(reading_type: str, value: float) -> Tuple[Severity, float]:
    """Return (severity, threshold_proximity 0..1) for a reading value."""
    cfg = THRESHOLDS.get(reading_type)
    if not cfg:
        return Severity.low, 0.0
    severity = Severity.low
    proximity = 0.0
    prev = 0.0
    for thresh, sev in cfg["levels"]:
        if value >= thresh:
            severity = Severity(sev)
            prev = thresh
        elif value >= thresh * 0.75:
            # approaching this threshold
            proximity = max(proximity, (value - prev) / max(thresh - prev, 1e-9) * 0.5)
            prev = thresh
    if value >= cfg["levels"][0][0]:
        # at/above at least first threshold — proximity based on distance to next
        thresholds = [t for t, _ in cfg["levels"]]
        for i, t in enumerate(thresholds):
            if value >= t:
                nxt = thresholds[i + 1] if i + 1 < len(thresholds) else t * 1.25
                proximity = max(proximity, 0.5 + 0.5 * min(1.0, (value - t) / (nxt - t)))
    return severity, proximity


def _trend(db: Session, site_id: int, reading_type: str, hours: int = 24) -> Optional[float]:
    """Slope of the last N hours of readings (units per hour). None if insufficient data."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    readings = (
        db.query(SensorReading)
        .filter(
            SensorReading.sensor_site_id == site_id,
            SensorReading.hazard_type == reading_type,
            SensorReading.timestamp >= cutoff,
        )
        .order_by(SensorReading.timestamp)
        .all()
    )
    if len(readings) < 3:
        return None
    t0 = readings[0].timestamp
    xs = np.array([(r.timestamp - t0).total_seconds() / 3600.0 for r in readings])
    ys = np.array([r.value for r in readings])
    if xs.std() == 0:
        return None
    slope = np.polyfit(xs, ys, 1)[0]
    return float(slope)


def _affected_polygon(center_lon: float, center_lat: float, radius_km: float) -> str:
    """Approximate circular affected area as a WKT polygon (simple planar offset)."""
    deg_lat = radius_km / 111.0
    deg_lon = radius_km / (111.0 * max(math.cos(math.radians(center_lat)), 0.1))
    pts = []
    for ang in range(0, 360, 30):
        rad = math.radians(ang)
        pts.append(f"{center_lon + deg_lon * math.cos(rad):.5f} {center_lat + deg_lat * math.sin(rad):.5f}")
    pts.append(pts[0])
    return f"POLYGON(({', '.join(pts)}))"


def generate_scenarios_from_live_data(db: Session, max_scenarios: int = 20) -> List[Scenario]:
    """
    Scan fresh live readings, build scenarios anchored on real values + trends.
    Monte Carlo samples magnitude in the plausible band around the live reading.
    """
    scenarios: List[Scenario] = []
    rng = random.Random(datetime.utcnow().microsecond)

    # candidate reading types we can build scenarios from
    candidate_types = ["earthquake", "rainfall", "flood", "wind_speed", "wind_gust", "temperature"]
    ranked = []

    for rtype in candidate_types:
        readings = (
            db.query(SensorReading)
            .filter(SensorReading.hazard_type == rtype)
            .order_by(SensorReading.timestamp.desc())
            .limit(50)
            .all()
        )
        if not readings:
            continue
        for reading in readings:
            severity, proximity = _severity_from_value(rtype, reading.value)
            trend = _trend(db, reading.sensor_site_id, rtype)
            # Only consider readings that are near/above a threshold or rising fast
            if severity == Severity.low and (trend is None or trend <= 0):
                continue
            # Probability from proximity + trend + severity
            prob = 0.05 + 0.35 * proximity
            if trend and trend > 0:
                prob += min(0.25, trend * 0.02)
            if severity == Severity.extreme:
                prob += 0.25
            elif severity == Severity.high:
                prob += 0.15
            elif severity == Severity.medium:
                prob += 0.08
            prob = min(0.95, prob)
            ranked.append((prob * ({"low": 1, "medium": 2, "high": 3, "extreme": 4}[severity.value]),
                           prob, severity, reading, trend))

    ranked.sort(key=lambda x: x[0], reverse=True)

    for score, prob, severity, reading, trend in ranked[:max_scenarios]:
        site = db.query(SensorSite).get(reading.sensor_site_id)
        rtype = reading.hazard_type.value if hasattr(reading.hazard_type, "value") else reading.hazard_type

        # Monte Carlo sample plausible peak value around the live reading + trend
        trend_component = (trend or 0.0) * 6.0  # project 6h ahead
        mean = max(reading.value + trend_component, reading.value)
        samples = rng.gauss(mean, max(0.15 * mean, 0.5))
        peak = max(samples, reading.value * 0.8)

        horizon = 24 if rtype != "earthquake" else 6
        area = _affected_polygon(site.longitude, site.latitude, radius_km=3 + 4 * prob)

        scenario = Scenario(
            hazard_type=reading.hazard_type,
            generated_from_reading_id=reading.id,
            probability=round(prob, 3),
            severity=severity,
            time_horizon_hours=horizon,
            affected_area=area,
            status=ScenarioStatus.complete,
        )
        db.add(scenario)
        db.flush()  # get scenario.id

        # --- Events: initial + peak + decay, then cascades ---
        events = []

        def add_event(etype, offset_h, magnitude, desc, parent_id=None):
            ev = ScenarioEvent(
                scenario_id=scenario.id,
                event_type=etype,
                location=f"POINT({site.longitude:.5f} {site.latitude:.5f})",
                timestamp_offset_hours=offset_h,
                magnitude=round(float(magnitude), 2),
                description=desc,
                cascade_parent_id=parent_id,
            )
            events.append(ev)

        base_ev = add_event("initial", 0.0, reading.value, f"Live reading {reading.value} {reading.unit} from {site.name} ({rtype})")
        peak_ev = add_event("peak", horizon * 0.3, peak, f"Modeled peak {peak:.1f} {reading.unit} (trend {trend if trend is None else round(trend, 3)}/h)")

        # Cascade chain from real rule table
        rules = CASCADE_RULES.get(rtype, [])
        parent = peak_ev
        for rule in rules:
            trigger = rule.get("min_magnitude") if rtype == "earthquake" else rule.get("min_value")
            if trigger is not None and peak >= trigger and rng.random() < rule["prob"] + 0.1 * prob:
                child = add_event("cascade_step", rule["offset_h"], trigger, rule["description"], parent_id=None)
                parent = child  # allow chaining one level

        scenarios.append(scenario)

    db.commit()
    return scenarios


def expire_old_scenarios(db: Session, max_age_hours: int = 72):
    """Mark scenarios older than max_age as expired so the UI stays honest."""
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    db.query(Scenario).filter(Scenario.created_at < cutoff, Scenario.status != ScenarioStatus.expired).update(
        {"status": ScenarioStatus.expired}, synchronize_session=False
    )
    db.commit()