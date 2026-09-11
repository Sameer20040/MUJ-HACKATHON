import httpx
from datetime import datetime, timezone
from typing import List, Optional
from .base import BaseConnector, ReadingDTO
import logging

logger = logging.getLogger(__name__)

class USGSEarthquakeConnector(BaseConnector):
    def __init__(self):
        super().__init__("usgs_earthquake")
        self.base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    async def fetch_site(self, site_id: int, site_config: dict) -> List[ReadingDTO]:
        """
        Fetch recent earthquake data from USGS.
        Note: USGS doesn't support site-specific queries easily, so we fetch a regional query
        and filter by proximity to the site.
        For simplicity in this example, we'll fetch global recent quakes and let the caller
        handle proximity filtering or we assume the site represents a region of interest.
        """
        readings = []
        latitude = site_config.get('latitude')
        longitude = site_config.get('longitude')

        if latitude is None or longitude is None:
            logger.warning(f"Site {site_id} missing latitude/longitude for USGS Earthquake")
            return readings

        # Create WKT point string for location
        location_wkt = f"POINT({longitude} {latitude})"

        try:
            # Parameters for the USGS API
            params = {
                'format': 'geojson',
                'limit': 50,  # Get recent events
                'minmagnitude': 2.5,  # Only significant quakes
                'orderby': 'time',
            }

            # Scope quakes to a radius around this site so each region gets its own
            # nearby events instead of one global list assigned to every site.
            params['latitude'] = latitude
            params['longitude'] = longitude
            params['maxradiuskm'] = 500

            # Calculate starttime for last 7 days (radius-scoped, so volume stays low)
            from datetime import timedelta
            start_time = datetime.now(timezone.utc) - timedelta(days=7)
            params['starttime'] = start_time.strftime('%Y-%m-%dT%H:%M:%S')

            data = await self.get_json(self.base_url, params=params)

            if 'features' not in data:
                logger.warning("No features in USGS response")
                return readings

            for feature in data['features']:
                props = feature['properties']
                geom = feature['geometry']

                if not geom or geom['type'] != 'Point':
                    continue

                # Extract earthquake details
                magnitude = props.get('mag')
                if magnitude is None:
                    continue

                # Use magnitude as the value (could also use intensity, etc.)
                eq_longitude, eq_latitude, depth = geom['coordinates']

                # Create a reading with the earthquake's location
                dto = ReadingDTO(
                    sensor_site_id=site_id,  # This is simplistic - ideally we'd match to nearest site
                    hazard_type="earthquake",
                    location=location_wkt,
                    value=float(magnitude),
                    unit="magnitude",
                    timestamp=datetime.fromtimestamp(props['time'] / 1000, tz=timezone.utc),
                    source=self.source_name,
                    raw_payload=props  # Store the raw properties for debugging
                )
                readings.append(dto)

        except Exception as e:
            logger.error(f"Error fetching USGS earthquake data: {type(e).__name__}: {e}")
            # Return empty list on error - the ingestor will handle this gracefully

        return readings