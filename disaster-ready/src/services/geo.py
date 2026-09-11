"""Geospatial + proximity helpers (SQLite/WKT-safe, PostGIS-style semantics).

Region is derived from coordinates (not stored) so the same function tags sites,
scenarios, tracked assets and official alerts consistently.
"""
import math
from typing import List, Optional, Tuple

from shapely import wkt as shapely_wkt
from shapely.geometry import Point

from src.models import Scenario, ScenarioStatus, OfficialAlert, ResourcePosition, EvacuationRoute

# Region bounding boxes: (lat_min, lat_max, lon_min, lon_max)
REGION_BBOXES = {
    "India": (6.5, 37.5, 68.0, 97.5),
    "United States": (24.0, 50.0, -125.0, -66.0),
}
ALL_REGIONS = ["India", "United States", "Global"]


def region_of(lat: Optional[float], lon: Optional[float]) -> str:
    """Bucket a coordinate into a named region, else 'Global'."""
    if lat is None or lon is None:
        return "Global"
    for name, (la_min, la_max, lo_min, lo_max) in REGION_BBOXES.items():
        if la_min <= lat <= la_max and lo_min <= lon <= lo_max:
            return name
    return "Global"


def region_center(region: str) -> Tuple[float, float]:
    """A sensible map center [lat, lon] for a region."""
    if region == "India":
        return (22.0, 79.0)
    if region == "United States":
        return (39.0, -98.0)
    return (20.0, 0.0)  # Global


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _load_geom(wkt_str: Optional[str]):
    if not wkt_str:
        return None
    try:
        return shapely_wkt.loads(wkt_str)
    except Exception:
        return None


def point_in_active_scenarios(db, lat: float, lon: float) -> List[Scenario]:
    """Active scenarios whose affected_area polygon contains the point."""
    pt = Point(lon, lat)
    hits = []
    scenarios = db.query(Scenario).filter(Scenario.status != ScenarioStatus.expired).all()
    for sc in scenarios:
        poly = _load_geom(sc.affected_area)
        if poly is not None and poly.contains(pt):
            hits.append(sc)
    return hits


def alerts_containing_point(db, lat: float, lon: float, radius_km: float = 50.0) -> List[OfficialAlert]:
    """Official alerts whose polygon contains the point, or whose point is within radius_km."""
    pt = Point(lon, lat)
    hits = []
    for al in db.query(OfficialAlert).all():
        geom = _load_geom(al.geometry)
        if geom is None:
            continue
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            if geom.contains(pt):
                hits.append(al)
        else:  # Point / other: use distance
            try:
                gy, gx = geom.centroid.y, geom.centroid.x
                if haversine_km(lat, lon, gy, gx) <= radius_km:
                    hits.append(al)
            except Exception:
                continue
    return hits


def nearest_resource(db, lat: float, lon: float, resource_type: Optional[str] = None):
    """Return (ResourcePosition, distance_km) nearest to the point, or (None, None)."""
    q = db.query(ResourcePosition)
    if resource_type:
        q = q.filter(ResourcePosition.resource_type == resource_type)
    best, best_d = None, None
    for res in q.all():
        geom = _load_geom(res.location)
        if geom is None:
            continue
        d = haversine_km(lat, lon, geom.y, geom.x)
        if best_d is None or d < best_d:
            best, best_d = res, d
    return best, best_d


def nearest_route(db, lat: float, lon: float):
    """Return (EvacuationRoute, distance_km) whose path passes nearest the point."""
    best, best_d = None, None
    for route in db.query(EvacuationRoute).all():
        line = _load_geom(route.path)
        if line is None:
            continue
        # min haversine to any vertex of the path (good enough for a 'nearest route' hint)
        try:
            coords = list(line.coords)
        except Exception:
            continue
        for x, y in coords:
            d = haversine_km(lat, lon, y, x)
            if best_d is None or d < best_d:
                best, best_d = route, d
    return best, best_d
