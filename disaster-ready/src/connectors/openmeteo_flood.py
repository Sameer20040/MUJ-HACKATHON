from datetime import datetime, timezone
from typing import List
import logging

from .base import BaseConnector, ReadingDTO

logger = logging.getLogger(__name__)


class OpenMeteoFloodConnector(BaseConnector):
    """Global river discharge from the Open-Meteo Flood API (keyless, worldwide).

    Used for India and other non-US sites where USGS Water Services has no coverage.
    """

    def __init__(self):
        super().__init__("openmeteo_flood")
        self.base_url = "https://flood-api.open-meteo.com/v1/flood"

    async def fetch_site(self, site_id: int, site_config: dict) -> List[ReadingDTO]:
        readings: List[ReadingDTO] = []
        latitude = site_config.get('latitude')
        longitude = site_config.get('longitude')

        if latitude is None or longitude is None:
            logger.warning(f"Site {site_id} missing latitude/longitude for Open-Meteo Flood")
            return readings

        location_wkt = f"POINT({longitude} {latitude})"

        try:
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'daily': 'river_discharge',
                'forecast_days': 1,
                'timezone': 'UTC',
            }
            data = await self.get_json(self.base_url, params=params)

            daily = data.get('daily') or {}
            times = daily.get('time') or []
            discharges = daily.get('river_discharge') or []
            if not times or not discharges:
                logger.warning(f"No river_discharge in Open-Meteo Flood response for site {site_id}")
                return readings

            # Take the most recent non-null value.
            value = None
            ts_str = None
            for t, v in zip(times, discharges):
                if v is not None:
                    value, ts_str = v, t
            if value is None:
                logger.warning(f"All river_discharge values null for site {site_id}")
                return readings

            unit = (data.get('daily_units') or {}).get('river_discharge', 'm³/s')
            try:
                timestamp = datetime.fromisoformat(ts_str)
            except Exception:
                timestamp = datetime.now(timezone.utc)

            readings.append(ReadingDTO(
                sensor_site_id=site_id,
                hazard_type="flood",
                location=location_wkt,
                value=float(value),
                unit=unit,
                timestamp=timestamp,
                source=self.source_name,
                raw_payload={'river_discharge': value, 'time': ts_str},
            ))

        except Exception as e:
            logger.error(f"Error fetching Open-Meteo Flood data for site {site_id}: {type(e).__name__}: {e}")

        return readings
