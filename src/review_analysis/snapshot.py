"""A snapshot is a directory holding one collection run, committed so results are reproducible.

<dir>/manifest.json       app info, parameters, per-storefront stats, warnings
<dir>/pool.jsonl          every review in the window, one JSON object per line
<dir>/samples/<name>.json sample definitions (strategy, n, seed, review_ids)
<dir>/raw/<cc>/*.json     the pages exactly as Apple returned them (resume cache + audit trail)
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from review_analysis.collector import PoolResult
from review_analysis.models import AppInfo, Review, StorefrontStats

SampleStrategy = Literal["random", "recent"]


class SampleDef(BaseModel):
    name: str
    strategy: SampleStrategy
    n_requested: int
    n: int
    seed: int | None
    pool_size: int
    window_start: datetime
    window_end: datetime
    review_ids: list[str]


class Manifest(BaseModel):
    app: AppInfo
    since: datetime
    collected_at: datetime
    storefronts: list[StorefrontStats]
    pool_size: int
    duplicates_removed: int
    warnings: list[str]
    collector_version: str
    params: dict[str, Any]


def write_snapshot(
    out_dir: Path,
    app: AppInfo,
    pool: PoolResult,
    samples: list[SampleDef],
    *,
    collector_version: str,
    params: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        app=app,
        since=pool.since,
        collected_at=pool.collected_at,
        storefronts=pool.storefronts,
        pool_size=len(pool.reviews),
        duplicates_removed=pool.duplicates_removed,
        warnings=pool.warnings,
        collector_version=collector_version,
        params=params,
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    with (out_dir / "pool.jsonl").open("w", encoding="utf-8") as fh:
        for review in pool.reviews:
            fh.write(review.model_dump_json() + "\n")
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    for sample in samples:
        (samples_dir / f"{sample.name}.json").write_text(
            sample.model_dump_json(indent=2), encoding="utf-8"
        )


def load_manifest(snapshot_dir: Path) -> Manifest:
    return Manifest.model_validate_json(
        (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
    )


def load_pool(snapshot_dir: Path) -> list[Review]:
    with (snapshot_dir / "pool.jsonl").open(encoding="utf-8") as fh:
        return [Review.model_validate_json(line) for line in fh if line.strip()]


def load_sample(snapshot_dir: Path, name: str) -> tuple[SampleDef, list[Review]]:
    sample = SampleDef.model_validate_json(
        (snapshot_dir / "samples" / f"{name}.json").read_text(encoding="utf-8")
    )
    by_id = {r.review_id: r for r in load_pool(snapshot_dir)}
    return sample, [by_id[i] for i in sample.review_ids]
