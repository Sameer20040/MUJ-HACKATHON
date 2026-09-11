import httpx
from datetime import datetime, timezone
from typing import List, Optional
from .base import BaseConnector, ReadingDTO
import logging

logger = logging.getLogger(__name__)

class OpenMeteoConnector(BaseConnector):
    def __init__(self):
        super().__init__("openmeteo")
        self.base_url = "https://api.open-meteo.com/v1/forecast"

    async def fetch_site(self, site_id: int, site_config: dict) -> List[ReadingDTO]:
        """
        Fetch weather data from Open-Meteo for a specific site.
        :param site_id: The ID of the site in our SensorSite table.
        :param site_config: Should contain 'latitude' and 'longitude' keys.
        :return: A list of ReadingDTO objects for various weather parameters.
        """
        readings = []
        latitude = site_config.get('latitude')
        longitude = site_config.get('longitude')

        if latitude is None or longitude is None:
            logger.warning(f"Site {site_id} missing latitude/longitude for Open-Meteo")
            return readings

        # Create WKT point string for location
        location_wkt = f"POINT({longitude} {latitude})"

        try:
            # We'll request multiple current weather variables
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'current': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_gusts_10m',
                'timezone': 'UTC'
            }

            data = await self.get_json(self.base_url, params=params)

            if 'current' not in data:
                logger.warning("No current data in Open-Meteo response")
                return readings

            current = data['current']
            timestamp_str = current.get('time')
            if not timestamp_str:
                logger.warning("No timestamp in Open-Meteo current data")
                return readings

            # Parse the timestamp (Open-Meteo returns ISO format string)
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

            # Create readings for each requested parameter
            # Temperature
            if 'temperature_2m' in current:
                dto = ReadingDTO(
                    sensor_site_id=site_id,
                    hazard_type="temperature",  # We'll map this appropriately later
                    location=location_wkt,
                    value=float(current['temperature_2m']),
                    unit="°C",
                    timestamp=timestamp,
                    source=self.source_name,
                    raw_payload={'temperature_2m': current['temperature_2m']}
                )
                readings.append(dto)

            # Relative Humidity
            if 'relative_humidity_2m' in current:
                dto = ReadingDTO(
                    sensor_site_id=site_id,
                    hazard_type="humidity",
                    location=location_wkt,
                    value=float(current['relative_humidity_2m']),
                    unit="%",
                    timestamp=timestamp,
                    source=self.source_name,
                    raw_payload={'relative_humidity_2m': current['relative_humidity_2m']}
                )
                readings.append(dto)

            # Precipitation
            if 'precipitation' in current:
                dto = ReadingDTO(
                    sensor_site_id=site_id,
                    hazard_type="rainfall",
                    location=location_wkt,
                    value=float(current['precipitation']),
                    unit="mm",
                    timestamp=timestamp,
                    source=self.source_name,
                    raw_payload={'precipitation': current['precipitation']}
                )
                readings.append(dto)

            # Wind Speed
            if 'wind_speed_10m' in current:
                dto = ReadingDTO(
                    sensor_site_id=site_id,
                    hazard_type="wind_speed",
                    location=location_wkt,
                    value=float(current['wind_speed_10m']),
                    unit="km/h",
                    timestamp=timestamp,
                    source=self.source_name,
                    raw_payload={'wind_speed_10m': current['wind_speed_10m']}
                )
                readings.append(dto)

            # Wind Gusts
            if 'wind_gusts_10m' in current:
                dto = ReadingDTO(
                    sensor_site_id=site_id,
                    hazard_type="wind_gust",
                    location=location_wkt,
                    value=float(current['wind_gusts_10m']),
                    unit="km/h",
                    timestamp=timestamp,
                    source=self.source_name,
                    raw_payload={'wind_gusts_10m': current['wind_gusts_10m']}
                )
                readings.append(dto)

        except Exception as e:
            logger.error(f"Error fetching Open-Meteo data for site {site_id}: {type(e).__name__}: {e}")

        return readings