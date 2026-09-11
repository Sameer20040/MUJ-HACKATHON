"""Export services for GeoJSON and PDF situation reports."""
import json
import logging
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

import sys
import importlib
import src.models
if not hasattr(src.models, 'TrackedAsset'):
    importlib.reload(src.models)

from shapely import wkt
from src.models import (
    Scenario, ScenarioStatus, TrackedAsset, AssetPosition,
    OfficialAlert, ResponsePlan, ResourcePosition, EvacuationRoute,
    SensorSite, SensorReading
)
from src.services.geo import region_of, region_center

logger = logging.getLogger(__name__)


def _clean_pdf_text(text: str) -> str:
    """Sanitize unicode strings to safe ASCII/Latin-1 for FPDF Helvetica font."""
    if not text:
        return ""
    replacements = {
        "—": "-", "–": "-", "•": "*", "…": "...",
        "“": '"', "”": '"', "‘": "'", "’": "'", "°": " deg "
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def build_geojson(
    db: Session,
    region: Optional[str] = None,
    include_assets: bool = True,
    include_alerts: bool = True,
    include_scenarios: bool = True,
    include_routes: bool = True,
    include_resources: bool = True,
    include_sensors: bool = True
) -> dict:
    """
    Build a GeoJSON FeatureCollection for the current situation.

    Args:
        db: Database session
        region: Optional region filter ("India", "United States", "Global", or None for all)
        include_assets: Include tracked assets
        include_alerts: Include official alerts
        include_scenarios: Include active scenarios
        include_routes: Include evacuation routes
        include_resources: Include resource positions
        include_sensors: Include sensor sites

    Returns:
        GeoJSON FeatureCollection dict
    """
    features = []

    try:
        # --- Active Scenarios ---
        if include_scenarios:
            scenarios = db.query(Scenario).filter(Scenario.status != ScenarioStatus.expired).all()
            for sc in scenarios:
                try:
                    # Check region filter
                    if region and region != "Global":
                        geom = wkt.loads(sc.affected_area)
                        centroid = geom.centroid
                        sc_region = region_of(centroid.y, centroid.x)
                        if sc_region != region:
                            continue

                    props = {
                        "id": f"scenario_{sc.id}",
                        "type": "scenario",
                        "hazard_type": sc.hazard_type.value if hasattr(sc.hazard_type, 'value') else str(sc.hazard_type),
                        "severity": sc.severity.value if hasattr(sc.severity, 'value') else str(sc.severity),
                        "probability": sc.probability,
                        "time_horizon_hours": sc.time_horizon_hours,
                        "created_at": sc.created_at.isoformat() if sc.created_at else None
                    }
                    features.append({
                        "type": "Feature",
                        "properties": props,
                        "geometry": json.loads(shapely_to_geojson(sc.affected_area))
                    })
                except Exception as e:
                    logger.debug(f"Skipping scenario {sc.id}: {e}")

        # --- Evacuation Routes ---
        if include_routes:
            plans = db.query(ResponsePlan).filter(ResponsePlan.is_active == True).all()
            for plan in plans:
                if region and region != "Global" and plan.region_name != region:
                    continue
                for route in plan.routes:
                    try:
                        props = {
                            "id": f"route_{route.id}",
                            "type": "evacuation_route",
                            "route_name": route.route_name,
                            "plan_name": plan.name,
                            "hazard_type": route.hazard_type.value if hasattr(route.hazard_type, 'value') else str(route.hazard_type),
                            "capacity_per_hour": route.capacity_per_hour,
                            "estimated_clearance_hours": route.estimated_clearance_hours
                        }
                        features.append({
                            "type": "Feature",
                            "properties": props,
                            "geometry": json.loads(shapely_to_geojson(route.path))
                        })
                    except Exception as e:
                        logger.debug(f"Skipping route {route.id}: {e}")

        # --- Resource Positions ---
        if include_resources:
            plans = db.query(ResponsePlan).filter(ResponsePlan.is_active == True).all()
            for plan in plans:
                if region and region != "Global" and plan.region_name != region:
                    continue
                for res in plan.resources:
                    try:
                        props = {
                            "id": f"resource_{res.id}",
                            "type": "resource_depot",
                            "resource_type": res.resource_type,
                            "plan_name": plan.name,
                            "quantity": res.quantity,
                            "capacity_per_hour": res.capacity_per_hour,
                            "deployment_time_hours": res.deployment_time_hours
                        }
                        features.append({
                            "type": "Feature",
                            "properties": props,
                            "geometry": json.loads(shapely_to_geojson(res.location))
                        })
                    except Exception as e:
                        logger.debug(f"Skipping resource {res.id}: {e}")

        # --- Sensor Sites ---
        if include_sensors:
            sites = db.query(SensorSite).filter(SensorSite.is_active == True).all()
            for site in sites:
                try:
                    # Check region filter
                    if region and region != "Global":
                        site_region = region_of(site.latitude, site.longitude)
                        if site_region != region:
                            continue

                    # Get latest reading
                    latest = db.query(SensorReading).filter(
                        SensorReading.sensor_site_id == site.id
                    ).order_by(SensorReading.timestamp.desc()).first()

                    props = {
                        "id": f"sensor_{site.id}",
                        "type": "sensor_site",
                        "name": site.name,
                        "hazard_type": site.hazard_type.value if hasattr(site.hazard_type, 'value') else str(site.hazard_type),
                        "source": site.source,
                        "is_active": site.is_active
                    }
                    if latest:
                        props["latest_value"] = latest.value
                        props["latest_unit"] = latest.unit
                        props["latest_timestamp"] = latest.timestamp.isoformat()

                    features.append({
                        "type": "Feature",
                        "properties": props,
                        "geometry": json.loads(shapely_to_geojson(site.location))
                    })
                except Exception as e:
                    logger.debug(f"Skipping sensor site {site.id}: {e}")

        # --- Tracked Assets ---
        if include_assets:
            assets = db.query(TrackedAsset).filter(TrackedAsset.is_active == True).all()
            for asset in assets:
                try:
                    # Check region filter
                    if region and region != "Global":
                        asset_region = asset.region or region_of(asset.latitude, asset.longitude)
                        if asset_region != region:
                            continue

                    # Current location
                    if asset.current_location:
                        props = {
                            "id": f"asset_{asset.id}",
                            "type": "tracked_asset",
                            "label": asset.label,
                            "asset_type": asset.asset_type,
                            "status": asset.status,
                            "speed_kmh": asset.speed_kmh,
                            "progress": asset.progress,
                            "plan_id": asset.plan_id,
                            "updated_at": asset.updated_at.isoformat() if asset.updated_at else None
                        }
                        features.append({
                            "type": "Feature",
                            "properties": props,
                            "geometry": json.loads(shapely_to_geojson(asset.current_location))
                        })

                    # Add path as a line feature if exists
                    if asset.path:
                        props = {
                            "id": f"asset_path_{asset.id}",
                            "type": "asset_path",
                            "label": asset.label,
                            "asset_type": asset.asset_type,
                            "asset_id": asset.id
                        }
                        features.append({
                            "type": "Feature",
                            "properties": props,
                            "geometry": json.loads(shapely_to_geojson(asset.path))
                        })

                except Exception as e:
                    logger.debug(f"Skipping asset {asset.id}: {e}")

        # --- Official Alerts ---
        if include_alerts:
            alerts = db.query(OfficialAlert).all()
            for alert in alerts:
                try:
                    # Check region filter
                    if region and region != "Global":
                        alert_region = alert.region
                        if alert_region != region:
                            continue

                    if alert.geometry:
                        props = {
                            "id": f"alert_{alert.id}",
                            "type": "official_alert",
                            "source": alert.source,
                            "event": alert.event,
                            "severity": alert.severity,
                            "headline": alert.headline,
                            "description": alert.description,
                            "area_desc": alert.area_desc,
                            "effective": alert.effective.isoformat() if alert.effective else None,
                            "expires": alert.expires.isoformat() if alert.expires else None,
                            "url": alert.url
                        }
                        features.append({
                            "type": "Feature",
                            "properties": props,
                            "geometry": json.loads(shapely_to_geojson(alert.geometry))
                        })
                except Exception as e:
                    logger.debug(f"Skipping alert {alert.id}: {e}")

    except Exception as e:
        logger.error(f"Error building GeoJSON: {e}")

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "generated_at": datetime.utcnow().isoformat(),
            "region": region or "All",
            "feature_count": len(features)
        }
    }


def build_situation_pdf(
    db: Session,
    region: Optional[str] = None,
    user_location: Optional[tuple] = None
) -> bytes:
    """
    Build a one-page PDF situation report.

    Args:
        db: Database session
        region: Optional region filter
        user_location: Optional (lat, lon) for user-specific report

    Returns:
        PDF bytes
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.error("fpdf2 not installed. Run: pip install fpdf2")
        raise

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 18)
    region_str = region or "Global"
    pdf.cell(0, 10, _clean_pdf_text(f"DisasterReady Situation Report - {region_str}"), ln=True, align="C")

    # Timestamp
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _clean_pdf_text(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"), ln=True, align="C")

    if user_location:
        pdf.cell(0, 6, _clean_pdf_text(f"User Location: {user_location[0]:.4f}, {user_location[1]:.4f}"), ln=True, align="C")

    pdf.ln(4)

    # --- Active Scenarios ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Active Scenarios", ln=True)
    pdf.set_font("Helvetica", "", 9)

    scenarios = db.query(Scenario).filter(Scenario.status != ScenarioStatus.expired).all()
    if region and region != "Global":
        filtered_scenarios = []
        for sc in scenarios:
            try:
                geom = wkt.loads(sc.affected_area)
                centroid = geom.centroid
                sc_region = region_of(centroid.y, centroid.x)
                if sc_region == region:
                    filtered_scenarios.append(sc)
            except Exception:
                pass
        scenarios = filtered_scenarios

    if scenarios:
        for sc in scenarios:
            sev = sc.severity.value if hasattr(sc.severity, 'value') else str(sc.severity)
            htype = sc.hazard_type.value if hasattr(sc.hazard_type, 'value') else str(sc.hazard_type)
            pdf.cell(0, 5, _clean_pdf_text(f"  * {htype.title()} - {sev.title()} (p={sc.probability:.0%}, horizon {sc.time_horizon_hours}h)"), ln=True)
    else:
        pdf.cell(0, 5, "  None", ln=True)

    pdf.ln(3)

    # --- Official Alerts ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Official Alerts (GDACS)", ln=True)
    pdf.set_font("Helvetica", "", 9)

    alerts = db.query(OfficialAlert).all()
    if region and region != "Global":
        alerts = [a for a in alerts if a.region == region]

    if alerts:
        for alert in alerts:
            pdf.cell(0, 5, _clean_pdf_text(f"  * {alert.event}: {alert.headline}"), ln=True)
            pdf.cell(0, 5, _clean_pdf_text(f"    Severity: {alert.severity} | Area: {alert.area_desc}"), ln=True)
            if alert.effective:
                pdf.cell(0, 5, _clean_pdf_text(f"    Effective: {alert.effective.strftime('%Y-%m-%d %H:%M')}"), ln=True)
    else:
        pdf.cell(0, 5, "  None", ln=True)

    pdf.ln(3)

    # --- Tracked Assets ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Tracked Assets", ln=True)
    pdf.set_font("Helvetica", "", 9)

    assets = db.query(TrackedAsset).filter(TrackedAsset.is_active == True).all()
    if region and region != "Global":
        assets = [a for a in assets if a.region == region]

    if assets:
        for asset in assets:
            pdf.cell(0, 5, _clean_pdf_text(f"  * {asset.label} ({asset.asset_type}) - {asset.status}"), ln=True)
            if asset.latitude and asset.longitude:
                pdf.cell(0, 5, _clean_pdf_text(f"    Position: {asset.latitude:.4f}, {asset.longitude:.4f} | Speed: {asset.speed_kmh} km/h | Progress: {asset.progress:.0%}"), ln=True)
    else:
        pdf.cell(0, 5, "  None", ln=True)

    pdf.ln(3)

    # --- Resources ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Resource Depots", ln=True)
    pdf.set_font("Helvetica", "", 9)

    plans = db.query(ResponsePlan).filter(ResponsePlan.is_active == True).all()
    if region and region != "Global":
        plans = [p for p in plans if p.region_name == region]

    resources_found = False
    for plan in plans:
        for res in plan.resources:
            try:
                geom = wkt.loads(res.location)
                res_region = region_of(geom.y, geom.x)
                if not region or region == "Global" or res_region == region:
                    pdf.cell(0, 5, _clean_pdf_text(f"  * {res.resource_type.replace('_', ' ').title()} x{res.quantity} - {plan.name}"), ln=True)
                    resources_found = True
            except Exception:
                pass

    if not resources_found:
        pdf.cell(0, 5, "  None", ln=True)

    pdf.ln(3)

    # --- Evacuation Routes ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Evacuation Routes", ln=True)
    pdf.set_font("Helvetica", "", 9)

    routes_found = False
    for plan in plans:
        for route in plan.routes:
            pdf.cell(0, 5, _clean_pdf_text(f"  * {route.route_name} - {route.capacity_per_hour or '?'} p/h"), ln=True)
            routes_found = True

    if not routes_found:
        pdf.cell(0, 5, "  None", ln=True)

    pdf.ln(5)

    # Footer
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "Data sources: USGS (earthquakes), Open-Meteo (weather/flood), GDACS (alerts).", ln=True, align="C")
    pdf.cell(0, 5, "Asset movement is SIMULATED for planning purposes only.", ln=True, align="C")

    return bytes(pdf.output())


def shapely_to_geojson(wkt_str: str) -> str:
    """Convert WKT string to GeoJSON geometry string."""
    from shapely import wkt
    import json as std_json

    try:
        geom = wkt.loads(wkt_str)
    except Exception:
        return std_json.dumps(None)

    geom_type = geom.geom_type

    if geom_type == "Point":
        x, y = geom.x, geom.y
        return std_json.dumps({"type": "Point", "coordinates": [x, y]})
    elif geom_type == "LineString":
        coords = list(geom.coords)
        return std_json.dumps({"type": "LineString", "coordinates": [[x, y] for x, y in coords]})
    elif geom_type == "Polygon":
        exterior = list(geom.exterior.coords)
        interiors = [list(ring.coords) for ring in geom.interiors]
        coords = [exterior] + interiors
        return std_json.dumps({"type": "Polygon", "coordinates": [[[x, y] for x, y in ring] for ring in coords]})
    elif geom_type == "MultiPolygon":
        polygons = []
        for poly in geom.geoms:
            exterior = list(poly.exterior.coords)
            interiors = [list(ring.coords) for ring in poly.interiors]
            coords = [exterior] + interiors
            polygons.append([[[x, y] for x, y in ring] for ring in coords])
        return std_json.dumps({"type": "MultiPolygon", "coordinates": polygons})
    else:
        return std_json.dumps(None)