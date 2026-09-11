"""Proximity alerting service for user location and asset tracking."""
import logging
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from datetime import datetime

import sys
import importlib
import src.models
if not hasattr(src.models, 'TrackedAsset'):
    importlib.reload(src.models)

from shapely import wkt
from src.models import OfficialAlert, TrackedAsset, AssetPosition, AlertSubscription
from src.services.geo import (
    haversine_km,
    point_in_active_scenarios,
    alerts_containing_point,
    nearest_resource,
    nearest_route
)

logger = logging.getLogger(__name__)


def evaluate_user_alerts(
    db: Session,
    lat: float,
    lon: float,
    radius_km: float = 25.0,
    min_severity: str = "medium"
) -> List[dict]:
    """
    Evaluate proximity alerts for a user location.

    Args:
        db: Database session
        lat: User latitude
        lon: User longitude
        radius_km: Alert radius in kilometers
        min_severity: Minimum severity level to alert on

    Returns:
        List of alert dictionaries with severity, message, distance, and recommended action
    """
    alerts = []
    severity_levels = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
    min_level = severity_levels.get(min_severity.lower(), 1)

    try:
        # Check for active scenarios containing the point
        scenarios = point_in_active_scenarios(db, lat, lon)
        for scenario in scenarios:
            if severity_levels.get(scenario.severity.value if hasattr(scenario.severity, 'value') else str(scenario.severity), 0) >= min_level:
                alerts.append({
                    "type": "scenario",
                    "severity": scenario.severity.value if hasattr(scenario.severity, 'value') else str(scenario.severity),
                    "message": f"You are inside a {scenario.hazard_type.value} hazard zone",
                    "distance_km": 0.0,
                    "recommended_action": "Seek shelter immediately and follow evacuation routes",
                    "scenario_id": scenario.id,
                    "hazard_type": scenario.hazard_type.value if hasattr(scenario.hazard_type, 'value') else str(scenario.hazard_type)
                })

        # Check for official alerts near the point
        official_alerts = alerts_containing_point(db, lat, lon, radius_km)
        for alert in official_alerts:
            alert_severity = alert.severity if alert.severity else "medium"
            if severity_levels.get(alert_severity.lower(), 0) >= min_level:
                distance_km = 0.0
                if alert.geometry:
                    try:
                        geom = wkt.loads(alert.geometry)
                        if geom.geom_type == "Point":
                            alert_lat, alert_lon = geom.y, geom.x
                            distance_km = haversine_km(lat, lon, alert_lat, alert_lon)
                    except Exception:
                        pass

                alerts.append({
                    "type": "official_alert",
                    "severity": alert_severity,
                    "message": f"{alert.event}: {alert.headline}",
                    "distance_km": distance_km,
                    "recommended_action": "Monitor situation and follow local authorities",
                    "alert_id": alert.id,
                    "event": alert.event,
                    "effective": alert.effective,
                    "expires": alert.expires
                })

        # Check proximity to tracked assets (warn if hazardous assets are nearby)
        assets = db.query(TrackedAsset).filter(TrackedAsset.is_active == True).all()
        for asset in assets:
            if asset.latitude is not None and asset.longitude is not None:
                distance_km = haversine_km(lat, lon, asset.latitude, asset.longitude)
                if distance_km <= radius_km and asset.status in ["en_route", "on_scene"]:
                    # Only alert if asset is carrying hazardous materials or similar concern
                    if asset.asset_type in ["fuel_truck", "chemical_tanker"]:
                        alerts.append({
                            "type": "hazardous_asset",
                            "severity": "medium",
                            "message": f"Hazardous asset ({asset.label}) within {distance_km:.1f} km",
                            "distance_km": distance_km,
                            "recommended_action": "Maintain distance and follow instructions",
                            "asset_id": asset.id,
                            "asset_label": asset.label,
                            "asset_type": asset.asset_type
                        })

        # Find nearest shelter and route for recommended actions
        nearest_shelter, shelter_dist = nearest_resource(db, lat, lon, "shelters")
        nearest_route_obj, route_dist = nearest_route(db, lat, lon)

        # Add general preparedness advice if no immediate threats but within warning radius
        if not alerts and shelter_dist and shelter_dist <= radius_km * 2:
            alerts.append({
                "type": "preparedness",
                "severity": "low",
                "message": f"No immediate threats detected. Nearest shelter {shelter_dist:.1f} km away",
                "distance_km": shelter_dist,
                "recommended_action": "Review evacuation plan and stay informed",
                "shelter_distance_km": shelter_dist
            })

        # Sort by severity (highest first) then by distance (closest first)
        alerts.sort(key=lambda x: (
            -severity_levels.get(x["severity"].lower(), 0),
            x["distance_km"]
        ))

        return alerts

    except Exception as e:
        logger.error(f"Error evaluating user alerts: {e}")
        return []


def send_email_alert(
    email: str,
    subject: str,
    message: str,
    smtp_host: str = "",
    smtp_port: int = 587,
    smtp_user: str = "",
    smtp_password: str = "",
    smtp_from: str = ""
) -> bool:
    """
    Send an email alert via SMTP.

    Args:
        email: Recipient email address
        subject: Email subject
        message: Email body
        smtp_host: SMTP server hostname
        smtp_port: SMTP server port
        smtp_user: SMTP username
        smtp_password: SMTP password
        smtp_from: Sender email address

    Returns:
        True if sent successfully, False otherwise
    """
    if not all([smtp_host, smtp_user, smtp_password, smtp_from]):
        logger.warning("SMTP not configured - email alert not sent")
        return False

    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        msg = MIMEMultipart()
        msg['From'] = smtp_from
        msg['To'] = email
        msg['Subject'] = subject

        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        text = msg.as_string()
        server.sendmail(smtp_from, email, text)
        server.quit()

        logger.info(f"Email alert sent to {email}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")
        return False


def check_subscriptions_and_send_alerts(db: Session) -> int:
    """
    Check all alert subscriptions and send notifications if needed.

    Args:
        db: Database session

    Returns:
        Number of alerts sent
    """
    alerts_sent = 0
    try:
        subscriptions = db.query(AlertSubscription).all()

        for sub in subscriptions:
            # This would typically be called with a specific location to check
            # For now, we'll just log that we checked the subscription
            logger.debug(f"Checked subscription for {sub.email} in region {sub.region}")

        # In a full implementation, this would:
        # 1. For each subscription, get a representative location for the region
        # 2. Call evaluate_user_alerts for that location
        # 3. Send email notifications for new/changed alerts
        # 4. Track which alerts have already been notified to avoid duplicates

    except Exception as e:
        logger.error(f"Error checking subscriptions: {e}")

    return alerts_sent