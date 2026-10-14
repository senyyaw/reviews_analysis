"""Collections = (snapshot directory, sample). A collection id is "<snapshot>.<sample>".

Every snapshot directory under the snapshots dir (with a manifest.json) contributes one "pool"
collection, holding every review collected in its window, plus one collection per file in
samples/. For the committed Notion snapshot that means notion_2026-10-13.pool,
notion_2026-10-13.A_random and notion_2026-10-13.B_recent.

Analysis results are computed on first request and cached as JSON next to the snapshot
(<snapshot>/cache/<sample>.<kind>.v<CACHE_VERSION>.json), so a read-only deployment can serve
them pre-computed.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from review_analysis.errors import ReviewAnalysisError
from review_analysis.snapshot import Manifest, SampleDef, load_manifest

# Bump when analysis code changes the results, so stale cached JSON is never served.
CACHE_VERSION = 1


class CollectionNotFoundError(ReviewAnalysisError):
    pass


class CollectionStore:
    def __init__(self, snapshots_dir: Path) -> None:
        self.root = snapshots_dir

    def snapshot_dirs(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(p for p in self.root.iterdir() if (p / "manifest.json").exists())

    def collection_ids(self) -> list[str]:
        out = []
        for snap in self.snapshot_dirs():
            out.append(f"{snap.name}.pool")
            out += [f"{snap.name}.{p.stem}" for p in sorted((snap / "samples").glob("*.json"))]
        return out

    def resolve(self, collection_id: str) -> tuple[Path, str]:
        name, _, sample = collection_id.rpartition(".")
        snap = self.root / name
        if not name or not (snap / "manifest.json").exists():
            raise CollectionNotFoundError(f"collection '{collection_id}' not found")
        if sample != "pool" and not (snap / "samples" / f"{sample}.json").exists():
            raise CollectionNotFoundError(f"collection '{collection_id}' not found")
        return snap, sample

    def manifest(self, snap: Path) -> Manifest:
        return load_manifest(snap)

    def sample_def(self, snap: Path, sample: str) -> SampleDef | None:
        if sample == "pool":
            return None
        return SampleDef.model_validate_json((snap / "samples" / f"{sample}.json").read_text())

    def analysed(self, snap: Path) -> bool:
        return (snap / "sentiment.jsonl").exists() and (snap / "processed.jsonl").exists()

    def cached(self, snap: Path, sample: str, kind: str, compute: Callable[[], Any]) -> Any:
        path = snap / "cache" / f"{sample}.{kind}.v{CACHE_VERSION}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        value = compute()
        try:
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(value, default=float), encoding="utf-8")
        except OSError:  # read-only deployment: serve without caching
            pass
        return value
