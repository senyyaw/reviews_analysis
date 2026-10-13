"""A polite, retrying HTTP client for Apple's endpoints.

Apple throttles aggressively (the web reviews endpoint answers 429 "API capacity exceeded" with no
Retry-After header), so every request goes through one place that enforces:

- a minimum interval between requests, plus jitter;
- exponential backoff with jitter on 429, 5xx, timeouts and connection errors;
- an overall time budget for the whole run, so a collection cannot hang forever.

Sleep, clock and randomness are injectable so the retry logic is testable without waiting.
"""

import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx

from review_analysis import __version__
from review_analysis.errors import BudgetExceededError, UpstreamError

log = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
USER_AGENT = f"review-analysis/{__version__} (research project; polite crawler)"


@dataclass(frozen=True)
class RetryPolicy:
    min_interval_s: float = 3.0  # between consecutive requests
    jitter_s: float = 1.0  # extra random delay added to every wait
    max_retries: int = 6
    backoff_base_s: float = 5.0  # first backoff; doubles each retry: 5, 10, 20, 40, 60, 60
    backoff_cap_s: float = 60.0
    timeout_s: float = 15.0  # per request
    budget_s: float = 30 * 60  # whole run


class PoliteClient:
    def __init__(
        self,
        policy: RetryPolicy | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self.policy = policy or RetryPolicy()
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._started = clock()
        self._last_request: float | None = None
        self.requests_made = 0
        self.retries_made = 0
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=self.policy.timeout_s,
            follow_redirects=True,
            transport=transport,
        )

    def __enter__(self) -> "PoliteClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get(self, url: str, params: Mapping[str, str | int] | None = None) -> httpx.Response:
        """GET with politeness and retries.

        Returns the response for any non-retryable status (including 4xx), so the caller decides
        what a 404 or 400 means. Raises UpstreamError when retries are exhausted.
        """
        last_problem = "unknown error"
        last_status: int | None = None
        for attempt in range(self.policy.max_retries + 1):
            if attempt > 0:
                self.retries_made += 1
                self._wait(self._backoff(attempt))
            self._respect_interval()
            try:
                self.requests_made += 1
                response = self._client.get(url, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_problem, last_status = f"{type(exc).__name__}: {exc}", None
                log.warning("request failed (%s), attempt %d", last_problem, attempt + 1)
                continue
            if response.status_code not in RETRYABLE_STATUS:
                return response
            last_problem, last_status = f"HTTP {response.status_code}", response.status_code
            log.info("HTTP %d from %s, attempt %d", response.status_code, url, attempt + 1)
        raise UpstreamError(
            f"giving up on {url} after {self.policy.max_retries + 1} attempts: {last_problem}",
            status_code=last_status,
        )

    def _backoff(self, attempt: int) -> float:
        base: float = self.policy.backoff_base_s * 2.0 ** (attempt - 1)
        return min(base, self.policy.backoff_cap_s)

    def _respect_interval(self) -> None:
        now = self._clock()
        if self._last_request is not None:
            elapsed = now - self._last_request
            if elapsed < self.policy.min_interval_s:
                self._wait(self.policy.min_interval_s - elapsed)
        self._last_request = self._clock()

    def _wait(self, seconds: float) -> None:
        seconds += self._rng.uniform(0, self.policy.jitter_s)
        if self._clock() + seconds - self._started > self.policy.budget_s:
            raise BudgetExceededError(
                f"time budget of {self.policy.budget_s:.0f}s exhausted "
                "(likely sustained throttling)"
            )
        self._sleep(seconds)
