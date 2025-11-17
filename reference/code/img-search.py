import os
import httpx
from enum import Enum
from typing import TypeAlias, Any, Dict, Optional
from functools import lru_cache
from logging import getLogger

logger = getLogger(__name__)


BRAVE_WEB_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_IMG_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/images/search"

class SearchError(Exception):
    """Base exception for search failures."""
    pass


class ConfigError(SearchError):
    """Raised when API configuration is missing or invalid."""
    pass


class APIRequestError(SearchError):
    """Raised on HTTP request failures."""
    def __init__(
            self,
            status: int,
            message: str, *,
            response: Optional[httpx.Response] = None
    ):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.response = response

class SearchAPI(Enum):
    Brave = "brave"
    Bing = "bing" # Bing API now defunct
    Serp = "serpapi" # TODO

class ImgSearch:
    default_headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }

    def __init__(self, engine: str | SearchAPI, query: str = ""):
        if isinstance(engine, str):
            try:
                engine = SearchAPI(engine.lower())
            except ValueError:
                raise ConfigError(f"Unknown search engine: {engine!r}")

        self.engine = engine
        self.query = query

    def _headers(self) -> Dict[str, str]:
        h = dict(self.default_headers) # copy defaults
        if self.engine is SearchAPI.Brave:
            api_key = os.getenv("BRAVE_SEARCH_API_KEY")
            if not api_key:
                raise ConfigError("Missing BRAVE_SEARCH_API_KEY environment variable.")
            h["X-Subscription-Token"] = api_key
        return h

    def _url(self) -> str:
        match self.engine:
            case SearchAPI.Brave:
                return BRAVE_IMG_SEARCH_ENDPOINT
            case _:
                raise NotImplementedError(f"No URL defined for {self.engine}")

    def _params(self) -> Dict[str, Any]:
        match self.engine:
            case SearchAPI.Brave:
                return {"q": self.query}
            case _:
                return {}

    def search(self) -> Dict[str, Any]:
        """
        Execute the image search.

        Returns:
            Dict containing search results from the API

        Raises:
            APIRequestError: On network, timeout, or HTTP errors
            ConfigError: On missing configuration
        """

        # TODO: Cache w/ Redis
        redis_key = os.getenv("REDIS_KEY")
        if redis_key:
            pass
        return self._search()

    def _search(self) -> Dict[str, Any]:
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(
                    self._url(),
                    params=self._params(),
                    headers=self._headers())

                r.raise_for_status()
                data = r.json()

                logger.debug("\n Search results:\n")
                logger.debug(data)
                return data
        except httpx.HTTPStatusError as e:
            raise APIRequestError(
                e.response.status_code,
                f"HTTP error: {e}",
                response=e.response
            ) from e
        except httpx.TimeoutException as e:
            raise APIRequestError(408, "Request timed out") from e
        except httpx.RequestError as e:
            raise APIRequestError(503, f"Network error: {e}") from e
        except ValueError as e:  # Covers JSONDecodeError
            raise APIRequestError(
                getattr(r, 'status_code', 0),
                f"Invalid JSON response: {e}"
            ) from e



