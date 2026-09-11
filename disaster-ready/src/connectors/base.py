import asyncio
import httpx
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Public hazard APIs are occasionally slow; 5s (httpx default) times out far too eagerly.
DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
MAX_ATTEMPTS = 3

class ReadingDTO(BaseModel):
    sensor_site_id: int
    hazard_type: str
    location: str  # WKT Point string
    value: float
    unit: str
    timestamp: datetime
    source: str
    raw_payload: Optional[dict] = None

class BaseConnector(ABC):
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.client = self._new_client()

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": "DisasterReady/1.0 (live hazard monitoring)"},
        )

    @abstractmethod
    async def fetch_site(self, site_id: int, site_config: dict) -> List[ReadingDTO]:
        """
        Fetch data for a specific site.
        :param site_id: The ID of the site in our SensorSite table.
        :param site_config: A dictionary containing the site's configuration (like latitude, longitude, etc.)
        :return: A list of ReadingDTO objects.
        """
        pass

    def _ensure_client(self):
        """Recreate the HTTP client if it was closed (runner closes connectors each cycle)."""
        if self.client is None or self.client.is_closed:
            self.client = self._new_client()

    async def get_json(self, url: str, *, params: Optional[dict] = None) -> dict:
        """Fetch JSON with bounded retries for transient network/API failures."""
        last_error = None
        for attempt in range(MAX_ATTEMPTS):
            self._ensure_client()
            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                # Only 5xx / 429 are worth retrying; 4xx client errors will not fix themselves.
                last_error = exc
                if exc.response.status_code not in (429, 500, 502, 503, 504):
                    raise
            except httpx.TransportError as exc:
                # Covers timeouts, network errors, and protocol errors (e.g. server
                # disconnecting without a response) — all transient, all retryable.
                last_error = exc
            if attempt + 1 < MAX_ATTEMPTS:
                await asyncio.sleep(2 ** attempt)
        raise last_error  # type: ignore[misc]

    async def close(self):
        if self.client and not self.client.is_closed:
            await self.client.aclose()