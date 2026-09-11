"""Tracking simulator for advancing assets along paths."""
import asyncio
import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timezone

import sys
import importlib
import src.models
if not hasattr(src.models, 'TrackedAsset'):
    importlib.reload(src.models)

from src.models import TrackedAsset, AssetPosition
from src.database import get_session_local
from src.services.geo import haversine_km
from shapely import wkt
from shapely.geometry import LineString, Point

logger = logging.getLogger(__name__)


async def advance_assets(db: Session, dt_seconds: float) -> int:
    """
    Advance all active tracked assets along their paths.

    Args:
        db: Database session
        dt_seconds: Time step in seconds

    Returns:
        Number of assets that reached their target (progress >= 1.0)
    """
    advanced_count = 0
    completed_count = 0

    try:
        # Get all active tracked assets with paths
        assets = db.query(TrackedAsset).filter(
            TrackedAsset.is_active == True,
            TrackedAsset.path.isnot(None),
            TrackedAsset.current_location.isnot(None)
        ).all()

        for asset in assets:
            if not asset.path or not asset.current_location:
                continue

            try:
                # Load the path as a LineString
                line = wkt.loads(asset.path)
                if not isinstance(line, LineString):
                    continue

                # Calculate path length in kilometers
                path_len_km = 0.0
                coords = list(line.coords)
                for i in range(len(coords) - 1):
                    pt1 = Point(coords[i][0], coords[i][1])
                    pt2 = Point(coords[i+1][0], coords[i+1][1])
                    path_len_km += haversine_km(pt1.y, pt1.x, pt2.y, pt2.x)

                if path_len_km <= 0:
                    continue

                # Calculate distance to advance in this time step
                # speed_kmh * (dt_seconds / 3600) = distance_km
                distance_km = asset.speed_kmh * (dt_seconds / 3600.0)

                # Advance progress along the path (0.0 to 1.0)
                progress_increment = distance_km / path_len_km
                asset.progress = min(1.0, asset.progress + progress_increment)

                # Update current location based on progress
                if asset.progress >= 1.0:
                    # Reached target - set to target location
                    asset.current_location = asset.target
                    asset.status = "on_scene"
                    completed_count += 1
                else:
                    # Interpolate position along the path
                    point = line.interpolate(asset.progress, normalized=True)
                    asset.current_location = f"POINT({point.x} {point.y})"
                    asset.status = "en_route"

                asset.updated_at = datetime.now(timezone.utc)
                advanced_count += 1

                # Record position history
                position = AssetPosition(
                    asset_id=asset.id,
                    location=asset.current_location,
                    status=asset.status,
                    timestamp=datetime.now(timezone.utc)
                )
                db.add(position)

            except Exception as e:
                logger.error(f"Error advancing asset {asset.id}: {e}")
                continue

        # Commit all changes
        if advanced_count > 0:
            db.commit()

        # Prune old positions (keep last 20 per asset)
        await _prune_old_positions(db)

        return completed_count

    except Exception as e:
        logger.error(f"Error in advance_assets: {e}")
        db.rollback()
        return 0


async def _prune_old_positions(db: Session, keep_last_n: int = 20):
    """Prune asset positions to keep only the last N per asset."""
    try:
        # Get all assets
        assets = db.query(TrackedAsset.id).all()

        for (asset_id,) in assets:
            # Count positions for this asset
            pos_count = db.query(AssetPosition).filter(
                AssetPosition.asset_id == asset_id
            ).count()

            if pos_count > keep_last_n:
                # Delete oldest positions, keeping the newest keep_last_n
                delete_count = pos_count - keep_last_n
                old_positions = db.query(AssetPosition).filter(
                    AssetPosition.asset_id == asset_id
                ).order_by(AssetPosition.timestamp.asc()).limit(delete_count).all()

                for pos in old_positions:
                    db.delete(pos)

        db.commit()
    except Exception as e:
        logger.error(f"Error pruning old positions: {e}")
        db.rollback()


async def dispatch_asset(
    db: Session,
    label: str,
    asset_type: str,
    origin: str,
    target: str,
    plan_id: Optional[int] = None,
    route_id: Optional[int] = None,
    assigned_scenario_id: Optional[int] = None,
    speed_kmh: float = 40.0,
    region: Optional[str] = None
) -> Optional[TrackedAsset]:
    """
    Dispatch a new tracked asset.

    Args:
        db: Database session
        label: Human-readable label for the asset
        asset_type: Type of asset (rescue_team, ambulance, etc.)
        origin: WKT Point for starting location
        target: WKT Point for destination
        plan_id: Optional FK to ResponsePlan
        route_id: Optional FK to EvacuationRoute
        assigned_scenario_id: Optional FK to Scenario
        speed_kmh: Speed in km/h
        region: Region string for filtering

    Returns:
        Created TrackedAsset or None if failed
    """
    try:
        # Validate WKT format for origin and target
        try:
            wkt.loads(origin)
            wkt.loads(target)
        except Exception:
            logger.error(f"Invalid WKT format for origin or target: {origin}, {target}")
            return None

        # Create the path as a straight line from origin to target
        # In a more sophisticated version, this could follow roads or routes
        path_wkt = f"LINESTRING({origin.split('POINT(')[1].rstrip(')')}, {target.split('POINT(')[1].rstrip(')')})"

        asset = TrackedAsset(
            plan_id=plan_id,
            route_id=route_id,
            assigned_scenario_id=assigned_scenario_id,
            region=region,
            label=label,
            asset_type=asset_type,
            status="staged",
            path=path_wkt,
            origin=origin,
            target=target,
            current_location=origin,  # Start at origin
            progress=0.0,
            speed_kmh=speed_kmh,
            is_active=True
        )

        db.add(asset)
        db.commit()
        db.refresh(asset)

        # Record initial position
        position = AssetPosition(
            asset_id=asset.id,
            location=asset.current_location,
            status=asset.status,
            timestamp=datetime.now(timezone.utc)
        )
        db.add(position)
        db.commit()

        logger.info(f"Dispatched asset {asset.id}: {label} ({asset_type})")
        return asset

    except Exception as e:
        logger.error(f"Error dispatching asset: {e}")
        db.rollback()
        return None


async def seed_tracked_assets(db: Session):
    """Seed initial tracked assets for demo purposes."""
    try:
        # Check if we already have tracked assets
        if db.query(TrackedAsset).count() > 0:
            logger.info("Tracked assets already seeded")
            return

        # Get the first active plan for seeding
        plan = db.query(ResponsePlan).filter(ResponsePlan.is_active == True).first()
        if not plan:
            logger.warning("No active plan found for seeding tracked assets")
            return

        # Seed a few assets for demonstration
        assets_to_seed = [
            {
                "label": "Rescue Team Alpha",
                "asset_type": "rescue_team",
                "origin": "POINT(-118.2600 34.0500)",  # LA area
                "target": "POINT(-118.2437 34.0522)",   # Near LA seismic station
                "speed_kmh": 30.0
            },
            {
                "label": "Medical Supply Truck",
                "asset_type": "supply_truck",
                "origin": "POINT(-122.4300 37.7800)",   # SF area
                "target": "POINT(-122.4194 37.7749)",   # Near SF seismic station
                "speed_kmh": 25.0
            }
        ]

        for asset_data in assets_to_seed:
            await dispatch_asset(
                db=db,
                label=asset_data["label"],
                asset_type=asset_data["asset_type"],
                origin=asset_data["origin"],
                target=asset_data["target"],
                plan_id=plan.id,
                speed_kmh=asset_data["speed_kmh"],
                region=None  # Will be derived from coordinates
            )

        logger.info(f"Seeded {len(assets_to_seed)} tracked assets")

    except Exception as e:
        logger.error(f"Error seeding tracked assets: {e}")
        db.rollback()


async def run_tracking_loop(interval: int = 5):
    """Run the tracking advancement loop."""
    from src.database import get_session_local
    logger.info(f"Starting tracking loop with {interval}s interval")

    while True:
        try:
            # Get a fresh session for this cycle
            SessionLocal = get_session_local()
            db = SessionLocal()
            try:
                # Advance assets by the interval time
                completed = await advance_assets(db, float(interval))
                if completed > 0:
                    logger.info(f"Tracking loop: {completed} assets reached their targets")
            finally:
                db.close()

            # Wait for the next interval
            await asyncio.sleep(interval)

        except Exception as e:
            logger.error(f"Error in tracking loop: {e}")
            await asyncio.sleep(interval)  # Continue despite errors


# Global instance for easy access
tracking_simulator = None