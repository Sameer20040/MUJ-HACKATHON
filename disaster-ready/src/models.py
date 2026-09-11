from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text, Enum
from sqlalchemy.orm import relationship
from src.database import Base
import enum
from datetime import datetime
from shapely import wkt


class HazardType(str, enum.Enum):
    earthquake = "earthquake"
    flood = "flood"
    wildfire = "wildfire"
    cyclone = "cyclone"
    landslide = "landslide"
    volcanic = "volcanic"
    tsunami = "tsunami"
    rainfall = "rainfall"
    temperature = "temperature"
    humidity = "humidity"
    wind_speed = "wind_speed"
    wind_gust = "wind_gust"
    weather = "weather"


class Severity(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    extreme = "extreme"


class ScenarioStatus(str, enum.Enum):
    simulating = "simulating"
    complete = "complete"
    expired = "expired"


class FeedStatus(Base):
    __tablename__ = "feed_status"

    id = Column(Integer, primary_key=True, index=True)
    feed_name = Column(String, nullable=False, unique=True)
    last_update = Column(DateTime, nullable=True)
    status = Column(String, nullable=True)
    message = Column(Text, nullable=True)
    record_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SensorSite(Base):
    __tablename__ = "sensor_sites"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    hazard_type = Column(Enum(HazardType), nullable=False)
    location = Column(Text, nullable=False)
    upstream_url_template = Column(String, nullable=True)
    source = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    site_code = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    readings = relationship(
        "SensorReading",
        back_populates="sensor_site",
        cascade="all, delete-orphan"
    )

    @property
    def latitude(self):
        if self.location:
            try:
                return wkt.loads(self.location).y
            except Exception:
                return None
        return None

    @property
    def longitude(self):
        if self.location:
            try:
                return wkt.loads(self.location).x
            except Exception:
                return None
        return None


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    sensor_site_id = Column(
        Integer,
        ForeignKey("sensor_sites.id"),
        nullable=False
    )
    hazard_type = Column(Enum(HazardType), nullable=False)
    location = Column(Text, nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    source = Column(String, nullable=False)
    raw_payload = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sensor_site = relationship(
        "SensorSite",
        back_populates="readings"
    )


class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, index=True)
    hazard_type = Column(Enum(HazardType), nullable=False)
    generated_from_reading_id = Column(
        Integer,
        ForeignKey("sensor_readings.id"),
        nullable=True
    )
    probability = Column(Float, nullable=False)
    severity = Column(Enum(Severity), nullable=False)
    time_horizon_hours = Column(Integer, nullable=False)
    affected_area = Column(Text, nullable=False)
    status = Column(
        Enum(ScenarioStatus),
        default=ScenarioStatus.simulating
    )
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    generating_reading = relationship("SensorReading")
    events = relationship(
        "ScenarioEvent",
        back_populates="scenario",
        cascade="all, delete-orphan"
    )
    stress_tests = relationship(
        "PlanStressTest",
        back_populates="scenario"
    )


class ScenarioEvent(Base):
    __tablename__ = "scenario_events"

    id = Column(Integer, primary_key=True, index=True)
    scenario_id = Column(
        Integer,
        ForeignKey("scenarios.id"),
        nullable=False
    )
    event_type = Column(String, nullable=False)
    location = Column(Text, nullable=False)
    timestamp_offset_hours = Column(Float, nullable=False)
    magnitude = Column(Float, nullable=False)
    description = Column(String, nullable=True)
    cascade_parent_id = Column(
        Integer,
        ForeignKey("scenario_events.id"),
        nullable=True
    )

    scenario = relationship(
        "Scenario",
        back_populates="events"
    )

    parent = relationship(
        "ScenarioEvent",
        remote_side=[id],
        backref="children"
    )


class ResponsePlan(Base):
    __tablename__ = "response_plans"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    hazard_type = Column(Enum(HazardType), nullable=False)
    region_name = Column(String, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    plan_data = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    is_active = Column(Boolean, default=True)
    stress_test_score = Column(Float, nullable=True)

    stress_tests = relationship(
        "PlanStressTest",
        back_populates="plan"
    )

    resources = relationship(
        "ResourcePosition",
        back_populates="plan",
        cascade="all, delete-orphan"
    )

    routes = relationship(
        "EvacuationRoute",
        back_populates="plan",
        cascade="all, delete-orphan"
    )


class PlanStressTest(Base):
    __tablename__ = "plan_stress_tests"

    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(
        Integer,
        ForeignKey("response_plans.id"),
        nullable=False
    )
    scenario_id = Column(
        Integer,
        ForeignKey("scenarios.id"),
        nullable=False
    )
    evacuations_success_rate = Column(Float, nullable=True)
    resource_coverage = Column(Float, nullable=True)
    time_to_adequate_response_hours = Column(Float, nullable=True)
    bottlenecks = Column(Text, nullable=True)
    cascading_failures_detected = Column(Text, nullable=True)
    overall_score = Column(Float, nullable=True)
    recommended_revision = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    plan = relationship(
        "ResponsePlan",
        back_populates="stress_tests"
    )

    scenario = relationship(
        "Scenario",
        back_populates="stress_tests"
    )


class ResourcePosition(Base):
    __tablename__ = "resource_positions"

    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(
        Integer,
        ForeignKey("response_plans.id"),
        nullable=False
    )
    resource_type = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    location = Column(Text, nullable=False)
    capacity_per_hour = Column(Float, nullable=True)
    deployment_time_hours = Column(Float, nullable=True)
    assigned_scenario_id = Column(
        Integer,
        ForeignKey("scenarios.id"),
        nullable=True
    )

    plan = relationship(
        "ResponsePlan",
        back_populates="resources"
    )

    assigned_scenario = relationship("Scenario")


class EvacuationRoute(Base):
    __tablename__ = "evacuation_routes"

    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(
        Integer,
        ForeignKey("response_plans.id"),
        nullable=False
    )
    route_name = Column(String, nullable=False)
    hazard_type = Column(Enum(HazardType), nullable=False)
    path = Column(Text, nullable=False)
    capacity_per_hour = Column(Integer, nullable=True)
    estimated_clearance_hours = Column(Float, nullable=True)
    alt_route_id = Column(
        Integer,
        ForeignKey("evacuation_routes.id"),
        nullable=True
    )
    vulnerable_segments = Column(Text, nullable=True)

    plan = relationship(
        "ResponsePlan",
        back_populates="routes"
    )

    alt_route = relationship(
        "EvacuationRoute",
        remote_side=[id],
        backref="alternatives"
    )


class TrackedAsset(Base):
    __tablename__ = "tracked_assets"

    id = Column(Integer, primary_key=True, index=True)

    plan_id = Column(
        Integer,
        ForeignKey("response_plans.id"),
        nullable=True
    )

    route_id = Column(
        Integer,
        ForeignKey("evacuation_routes.id"),
        nullable=True
    )

    assigned_scenario_id = Column(
        Integer,
        ForeignKey("scenarios.id"),
        nullable=True
    )

    region = Column(String, nullable=True, index=True)

    label = Column(String, nullable=False)

    asset_type = Column(
        String,
        nullable=False
    )

    status = Column(
        String,
        nullable=False,
        default="staged"
    )

    path = Column(
        Text,
        nullable=True
    )

    origin = Column(
        Text,
        nullable=True
    )

    target = Column(
        Text,
        nullable=True
    )

    current_location = Column(
        Text,
        nullable=True
    )

    progress = Column(
        Float,
        nullable=False,
        default=0.0
    )

    speed_kmh = Column(
        Float,
        nullable=False,
        default=40.0
    )

    is_active = Column(
        Boolean,
        default=True
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    plan = relationship("ResponsePlan")

    route = relationship("EvacuationRoute")

    assigned_scenario = relationship("Scenario")

    positions = relationship(
        "AssetPosition",
        back_populates="asset",
        cascade="all, delete-orphan"
    )

    @property
    def latitude(self):
        if self.current_location:
            try:
                return wkt.loads(self.current_location).y
            except Exception:
                return None
        return None

    @property
    def longitude(self):
        if self.current_location:
            try:
                return wkt.loads(self.current_location).x
            except Exception:
                return None
        return None


class AssetPosition(Base):
    __tablename__ = "asset_positions"

    id = Column(Integer, primary_key=True, index=True)

    asset_id = Column(
        Integer,
        ForeignKey("tracked_assets.id"),
        nullable=False
    )

    location = Column(
        Text,
        nullable=False
    )

    status = Column(
        String,
        nullable=True
    )

    timestamp = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True
    )

    asset = relationship(
        "TrackedAsset",
        back_populates="positions"
    )


class OfficialAlert(Base):
    __tablename__ = "official_alerts"

    id = Column(Integer, primary_key=True, index=True)

    source = Column(
        String,
        nullable=False
    )

    external_id = Column(
        String,
        nullable=False,
        unique=True
    )

    event = Column(
        String,
        nullable=True
    )

    severity = Column(
        String,
        nullable=True
    )

    headline = Column(
        String,
        nullable=True
    )

    description = Column(
        Text,
        nullable=True
    )

    area_desc = Column(
        String,
        nullable=True
    )

    region = Column(
        String,
        nullable=True,
        index=True
    )

    effective = Column(
        DateTime,
        nullable=True
    )

    expires = Column(
        DateTime,
        nullable=True
    )

    geometry = Column(
        Text,
        nullable=True
    )

    url = Column(
        String,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


class AlertSubscription(Base):
    __tablename__ = "alert_subscriptions"

    id = Column(Integer, primary_key=True, index=True)

    email = Column(
        String,
        nullable=False
    )

    region = Column(
        String,
        nullable=True
    )

    min_severity = Column(
        String,
        nullable=True,
        default="medium"
    )

    radius_km = Column(
        Float,
        nullable=True,
        default=25.0
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )