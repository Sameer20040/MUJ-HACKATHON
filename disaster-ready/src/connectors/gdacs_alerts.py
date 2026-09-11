from datetime import datetime
from typing import List, Optional
import logging

from pydantic import BaseModel

from .base import BaseConnector
from src.services.geo import region_of

logger = logging.getLogger(__name__)

# GDACS event type codes -> human labels
EVENT_LABELS = {
    "EQ": "Earthquake", "TC": "Tropical Cyclone", "FL": "Flood",
    "VO": "Volcano", "DR": "Drought", "WF": "Wildfire", "TS": "Tsunami",
}
# GDACS alert levels -> our severity scale
ALERT_SEVERITY = {"green": "low", "orange": "high", "red": "extreme"}


class AlertDTO(BaseModel):
    source: str
    external_id: str
    event: Optional[str] = None
    severity: Optional[str] = None
    headline: Optional[str] = None
    description: Optional[str] = None
    area_desc: Optional[str] = None
    region: Optional[str] = None
    effective: Optional[datetime] = None
    expires: Optional[datetime] = None
    geometry: Optional[str] = None  # WKT
    url: Optional[str] = None


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


class GDACSAlertsConnector(BaseConnector):
    """Global disaster alerts from GDACS (keyless, worldwide incl. India).

    Unlike the sensor connectors this does not fetch per-site; it returns a global
    list of AlertDTOs that the runner upserts into OfficialAlert.
    """

    def __init__(self):
        super().__init__("gdacs_alerts")
        self.base_url = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"

    async def fetch_site(self, site_id: int, site_config: dict):
        """Not site-based; alerts are global. See fetch_alerts()."""
        return []

    async def fetch_alerts(self) -> List[AlertDTO]:
        alerts: List[AlertDTO] = []
        try:
            params = {"eventlist": "EQ;TC;FL;VO;DR"}
            data = await self.get_json(self.base_url, params=params)
            features = data.get("features", []) if isinstance(data, dict) else []

            for feat in features:
                props = feat.get("properties", {}) or {}
                geom = feat.get("geometry") or {}
                event_id = props.get("eventid")
                if event_id is None:
                    continue

                lon = lat = None
                wkt_geom = None
                if geom.get("type") == "Point":
                    coords = geom.get("coordinates") or []
                    if len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                        wkt_geom = f"POINT({lon} {lat})"

                level = str(props.get("alertlevel", "")).lower()
                event_code = props.get("eventtype", "")
                alerts.append(AlertDTO(
                    source="gdacs",
                    external_id=str(event_id),
                    event=EVENT_LABELS.get(event_code, event_code or "Event"),
                    severity=ALERT_SEVERITY.get(level, "medium"),
                    headline=props.get("name") or props.get("eventname"),
                    description=props.get("htmldescription") or props.get("description"),
                    area_desc=props.get("country"),
                    region=region_of(lat, lon),
                    effective=_parse_dt(props.get("fromdate")),
                    expires=_parse_dt(props.get("todate")),
                    geometry=wkt_geom,
                    url=f"https://www.gdacs.org/report.aspx?eventid={event_id}&eventtype={event_code}",
                ))

        except Exception as e:
            logger.error(f"Error fetching GDACS alerts: {type(e).__name__}: {e}")

        return alerts
