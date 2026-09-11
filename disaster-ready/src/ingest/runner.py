import asyncio
import logging
from datetime import datetime
from typing import Dict, List
from sqlalchemy.orm import Session

from src.connectors.usgs_earthquake import USGSEarthquakeConnector
from src.connectors.openmeteo import OpenMeteoConnector
from src.connectors.usgs_waterservice import USGSWaterServiceConnector
from src.connectors.openmeteo_flood import OpenMeteoFloodConnector
from src.connectors.gdacs_alerts import GDACSAlertsConnector
from src.models import FeedStatus, SensorReading, SensorSite, OfficialAlert
from src.database import get_session_local
from src.config import settings

logger = logging.getLogger(__name__)

class IngestRunner:
    def __init__(self):
        self.connectors = {
            'usgs_earthquake': USGSEarthquakeConnector(),
            'openmeteo': OpenMeteoConnector(),
            'usgs_waterservice': USGSWaterServiceConnector(),
            'openmeteo_flood': OpenMeteoFloodConnector(),
        }
        self.gdacs = GDACSAlertsConnector()  # not site-based; handled separately
        self.running = False
        self.session_local = get_session_local()

    async def update_feed_status(self, db: Session, feed_name: str, status: str,
                               message: str = None, record_count: int = None):
        """Update or create feed status record."""
        feed_status = db.query(FeedStatus).filter(FeedStatus.feed_name == feed_name).first()
        if not feed_status:
            feed_status = FeedStatus(feed_name=feed_name)
            db.add(feed_status)

        feed_status.status = status
        feed_status.last_update = datetime.utcnow()
        if message is not None:
            feed_status.message = message
        if record_count is not None:
            feed_status.record_count = record_count

        db.commit()

    async def run_ingestion_cycle(self, only_connector: str = None):
        """Run one cycle of data ingestion.

        :param only_connector: if set, only process sites for this connector/source.
            Each per-feed loop passes its own name so feeds honor their own cadence
            instead of every loop re-fetching every feed.
        """
        logger.info(f"Starting ingestion cycle ({only_connector or 'all feeds'})")

        # Get a fresh session for this cycle
        db = self.session_local()
        try:
            # Get all active sensor sites
            query = db.query(SensorSite).filter(SensorSite.is_active == True)
            if only_connector:
                query = query.filter(SensorSite.source == only_connector)
            sites = query.all()
            logger.info(f"Found {len(sites)} active sensor sites")

            # Group sites by connector type
            sites_by_connector: Dict[str, List[SensorSite]] = {}
            for site in sites:
                if site.source not in sites_by_connector:
                    sites_by_connector[site.source] = []
                sites_by_connector[site.source].append(site)

            # Process each connector type
            for connector_name, site_list in sites_by_connector.items():
                if connector_name not in self.connectors:
                    logger.warning(f"No connector found for source: {connector_name}")
                    await self.update_feed_status(db, connector_name, 'error',
                                              f"No connector found for source: {connector_name}")
                    continue

                connector = self.connectors[connector_name]
                try:
                    all_readings = []
                    for site in site_list:
                        # Prepare site config for the connector
                        site_config = {
                            'latitude': site.latitude,
                            'longitude': site.longitude,
                            'site_code': getattr(site, 'site_code', None)  # For waterservice
                        }

                        # Fetch data for this site
                        readings = await connector.fetch_site(site.id, site_config)
                        all_readings.extend(readings)
                        logger.debug(f"Fetched {len(readings)} readings from site {site.name}")

                    # Save readings to database
                    if all_readings:
                        for reading_dto in all_readings:
                            # Check if we already have this reading (by site and timestamp)
                            existing = db.query(SensorReading).filter(
                                SensorReading.sensor_site_id == reading_dto.sensor_site_id,
                                SensorReading.timestamp == reading_dto.timestamp
                            ).first()

                            if not existing:
                                new_reading = SensorReading(
                                    sensor_site_id=reading_dto.sensor_site_id,
                                    hazard_type=reading_dto.hazard_type,
                                    location=reading_dto.location,  # This should be WKT
                                    value=reading_dto.value,
                                    unit=reading_dto.unit,
                                    timestamp=reading_dto.timestamp,
                                    source=reading_dto.source,
                                    raw_payload=str(reading_dto.raw_payload) if reading_dto.raw_payload else None
                                )
                                db.add(new_reading)

                        db.commit()
                        logger.info(f"Saved {len(all_readings)} new readings from {connector_name}")
                        await self.update_feed_status(db, connector_name, 'ok',
                                                  f"Successfully ingested {len(all_readings)} readings",
                                                  len(all_readings))
                    else:
                        logger.warning(f"No readings fetched from {connector_name}")
                        await self.update_feed_status(db, connector_name, 'degraded',
                                                  "No readings fetched", 0)

                except Exception as e:
                    logger.error(f"Error in {connector_name} connector: {e}")
                    await self.update_feed_status(db, connector_name, 'error',
                                              f"Connector error: {str(e)}")
                finally:
                    await connector.close()

        except Exception as e:
            logger.error(f"Error in ingestion cycle: {e}")
            db.rollback()
        finally:
            db.close()

        logger.info("Finished ingestion cycle")

    async def run_alerts_cycle(self):
        """Fetch global GDACS official alerts and upsert into OfficialAlert."""
        logger.info("Starting GDACS alerts cycle")
        db = self.session_local()
        try:
            alerts = await self.gdacs.fetch_alerts()
            upserted = 0
            for dto in alerts:
                existing = db.query(OfficialAlert).filter(
                    OfficialAlert.external_id == dto.external_id
                ).first()
                if existing:
                    existing.severity = dto.severity
                    existing.headline = dto.headline
                    existing.description = dto.description
                    existing.area_desc = dto.area_desc
                    existing.region = dto.region
                    existing.effective = dto.effective
                    existing.expires = dto.expires
                    existing.geometry = dto.geometry
                    existing.url = dto.url
                else:
                    db.add(OfficialAlert(
                        source=dto.source, external_id=dto.external_id, event=dto.event,
                        severity=dto.severity, headline=dto.headline, description=dto.description,
                        area_desc=dto.area_desc, region=dto.region, effective=dto.effective,
                        expires=dto.expires, geometry=dto.geometry, url=dto.url,
                    ))
                    upserted += 1
            db.commit()
            status = 'ok' if alerts else 'degraded'
            await self.update_feed_status(db, 'gdacs_alerts', status,
                                          f"{len(alerts)} active alerts ({upserted} new)", len(alerts))
            logger.info(f"GDACS alerts cycle: {len(alerts)} alerts, {upserted} new")
        except Exception as e:
            logger.error(f"Error in GDACS alerts cycle: {type(e).__name__}: {e}")
            await self.update_feed_status(db, 'gdacs_alerts', 'error', f"Alerts error: {e}")
            db.rollback()
        finally:
            await self.gdacs.close()
            db.close()

    async def _run_alerts_loop(self, cadence: int):
        """Run the GDACS alerts cycle on a loop."""
        logger.info(f"Starting gdacs_alerts loop with {cadence}s cadence")
        while self.running:
            start_time = datetime.utcnow()
            await self.run_alerts_cycle()
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            sleep_time = max(0, cadence - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def start(self, cadence_override: Dict[str, int] = None):
        """Start the ingestion runner with specified cadences."""
        if self.running:
            logger.warning("Ingest runner is already running")
            return

        self.running = True
        logger.info("Starting ingest runner")

        # Use override cadences or defaults from settings
        cadences = {
            'usgs_earthquake': cadence_override.get('usgs_earthquake', settings.EARTHQUAKE_CADENCE) if cadence_override else settings.EARTHQUAKE_CADENCE,
            'openmeteo': cadence_override.get('openmeteo', settings.OPEN_METEO_CADENCE) if cadence_override else settings.OPEN_METEO_CADENCE,
            'usgs_waterservice': cadence_override.get('usgs_waterservice', settings.WATERSERVICE_CADENCE) if cadence_override else settings.WATERSERVICE_CADENCE,
            'openmeteo_flood': cadence_override.get('openmeteo_flood', settings.FLOOD_CADENCE) if cadence_override else settings.FLOOD_CADENCE,
        }

        # Create tasks for each connector with their respective cadences
        tasks = []
        for connector_name, cadence in cadences.items():
            task = asyncio.create_task(
                self._run_connector_loop(connector_name, cadence)
            )
            tasks.append(task)

        # Global official alerts (GDACS) run on their own loop, not per-site
        tasks.append(asyncio.create_task(self._run_alerts_loop(settings.GDACS_CADENCE)))

        # Wait for all tasks to complete (they run indefinitely until stopped)
        await asyncio.gather(*tasks)

    async def _run_connector_loop(self, connector_name: str, cadence: int):
        """Run a specific connector on a loop with given cadence."""
        logger.info(f"Starting {connector_name} loop with {cadence}s cadence")
        while self.running:
            start_time = datetime.utcnow()
            await self.run_ingestion_cycle(only_connector=connector_name)
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            sleep_time = max(0, cadence - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def stop(self):
        """Stop the ingestion runner."""
        logger.info("Stopping ingest runner")
        self.running = False

# Global instance
ingest_runner = IngestRunner()