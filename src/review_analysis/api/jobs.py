"""Background collection jobs.

Collecting an app's reviews takes minutes (Apple throttles to ~1 request every 3-5 s), which is
too long for one HTTP request. So POST /collections returns 202 Accepted with a job id, and
GET /jobs/{id} reports progress. Jobs run one at a time in a worker thread (a single process,
since models are loaded once), and they live in memory: a restart forgets job records, but every
finished collection is a snapshot on disk.
"""

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from review_analysis.errors import ReviewAnalysisError

JobState = Literal["queued", "running", "done", "failed"]


class Job(BaseModel):
    job_id: str
    state: JobState = "queued"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    app_id: int
    params: dict[str, Any]
    progress: list[str] = Field(default_factory=list)
    collection_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class JobManager:
    def __init__(self, executor: Executor | None = None) -> None:
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="collect")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for k, v in changes.items():
                setattr(job, k, v)
            job.updated_at = datetime.now(UTC)

    def _log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.progress = [*job.progress[-49:], message]
            job.updated_at = datetime.now(UTC)

    def submit(
        self,
        app_id: int,
        params: dict[str, Any],
        work: Callable[[Callable[[str], None]], list[str]],
    ) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], app_id=app_id, params=params)
        with self._lock:
            self._jobs[job.job_id] = job

        def run() -> None:
            self._update(job.job_id, state="running")
            try:
                ids = work(lambda msg: self._log(job.job_id, msg))
                self._update(job.job_id, state="done", collection_ids=ids)
            except ReviewAnalysisError as exc:
                self._update(job.job_id, state="failed", error=str(exc))
            except Exception as exc:
                self._update(job.job_id, state="failed", error=f"{type(exc).__name__}: {exc}")

        self._executor.submit(run)
        return self.get(job.job_id) or job
