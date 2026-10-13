import json
import random
from pathlib import Path
from typing import Any

import pytest

from review_analysis.http import PoliteClient, RetryPolicy

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeClock:
    """Monotonic clock that only advances when the client 'sleeps', so tests never wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_client(clock: FakeClock):  # type: ignore[no-untyped-def]
    def _make(**policy: Any) -> PoliteClient:
        defaults: dict[str, Any] = {"min_interval_s": 0.0, "jitter_s": 0.0}
        return PoliteClient(
            RetryPolicy(**(defaults | policy)),
            sleep=clock.sleep,
            clock=clock,
            rng=random.Random(0),
        )

    return _make
