from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Union


class Settings(BaseSettings):
    # Application
    app_name: str = "DisasterReady"
    environment: str = "dev"
    log_level: str = "INFO"

    # Database
    DATABASE_URL: Union[str, PostgresDsn]

    # Optional API keys
    NASA_FIRMS_KEY: str = ""
    USGS_VOLCANO_KEY: str = ""

    # Ingestion cadences (seconds)
    EARTHQUAKE_CADENCE: int = 300
    OPEN_METEO_CADENCE: int = 600
    WATERSERVICE_CADENCE: int = 900
    FLOOD_CADENCE: int = 900
    GDACS_CADENCE: int = 600
    FIRMS_CADENCE: int = 1800
    VOLCANO_CADENCE: int = 3600

    # Proximity alerting
    ALERT_RADIUS_KM: float = 25.0
    ALERT_MIN_SEVERITY: str = "medium"

    # Optional email (SMTP) for proximity alerts — degrades to in-app-only if unset
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()