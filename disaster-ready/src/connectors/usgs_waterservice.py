import httpx
from datetime import datetime, timezone
from typing import List, Optional
from .base import BaseConnector, ReadingDTO
import logging

logger = logging.getLogger(__name__)

class USGSWaterServiceConnector(BaseConnector):
    def __init__(self):
        super().__init__("usgs_waterservice")
        self.base_url = "https://waterservices.usgs.gov/nwis/iv/"

    async def fetch_site(self, site_id: int, site_config: dict) -> List[ReadingDTO]:
        """
        Fetch instantaneous water level/flow data from USGS Waterservice.
        :param site_id: The ID of the site in our SensorSite table.
        :param site_config: Should contain 'site_code' key (the USGS site code).
        :return: A list of ReadingDTO objects.
        """
        readings = []
        site_code = site_config.get('site_code')
        latitude = site_config.get('latitude')
        longitude = site_config.get('longitude')

        if not site_code:
            logger.warning(f"Site {site_id} missing site_code for USGS Waterservice")
            return readings

        # Create WKT point string for location
        location_wkt = f"POINT({longitude} {latitude})" if latitude and longitude else f"POINT(0 0)"

        try:
            # Parameters for the USGS Waterservice API
            params = {
                'format': 'json',
                'sites': site_code,
                'parameterCd': '00065',  # Gage height (stage) - most common for river level
                # Optional: '00060' for discharge (flow rate)
            }

            data = await self.get_json(self.base_url, params=params)

            if 'value' not in data or 'timeSeries' not in data['value']:
                logger.warning(f"No timeSeries data in USGS Waterservice response for site {site_code}")
                return readings

            time_series = data['value']['timeSeries']
            if not time_series:
                logger.warning(f"Empty timeSeries for site {site_code}")
                return readings

            # We'll take the first time series (there might be multiple for different parameters)
            ts = time_series[0]

            # Get the values
            if 'values' not in ts or not ts['values'] or 'value' not in ts['values'][0]:
                logger.warning(f"No values in time series for site {site_code}")
                return readings

            values = ts['values'][0]['value']
            if not values:
                logger.warning(f"Empty value array for site {site_code}")
                return readings

            # Take the most recent reading
            latest = values[-1]  # Values are in chronological order

            # Extract the value and timestamp
            value_str = latest.get('value')
            if value_str is None or value_str == '':
                logger.warning(f"Empty value in latest reading for site {site_code}")
                return readings

            try:
                value = float(value_str)
            except ValueError:
                logger.warning(f"Could not convert value '{value_str}' to float for site {site_code}")
                return readings

            # Parse the timestamp
            date_time_str = latest.get('dateTime')
            if not date_time_str:
                logger.warning(f"No dateTime in latest reading for site {site_code}")
                return readings

            # USGS Waterservice returns ISO 8601 timestamp
            timestamp = datetime.fromisoformat(date_time_str.replace('Z', '+00:00'))

            # Get unit info from the variable description
            unit = ts['variable']['unit']['unitCode']  # e.g., 'ft' for feet, 'm' for meters

            # Determine hazard type based on what we're measuring
            # For simplicity, we'll classify gage height as flood-related
            hazard_type = "flood"  # Could be made more sophisticated based on parameterCd

            dto = ReadingDTO(
                sensor_site_id=site_id,
                hazard_type=hazard_type,
                location=location_wkt,
                value=value,
                unit=unit,
                timestamp=timestamp,
                source=self.source_name,
                raw_payload=latest  # Store the raw value object for debugging
            )
            readings.append(dto)

        except Exception as e:
            logger.error(f"Error fetching USGS Waterservice data for site {site_code}: {type(e).__name__}: {e}")

        return readings