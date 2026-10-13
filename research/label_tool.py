"""Blind labelling tool for the human subset of the D1 test set.

    uv run python research/label_tool.py            # then open http://127.0.0.1:8765
    uv run python research/label_tool.py --complaint-pass 8766
        # re-check only reviews labelled negative with complaint=no (sentiment stays as labelled)

Shows title + text (plus the English translation for non-English reviews), never the star rating.
Reviews come in a fixed shuffled order, so the star strata are mixed. Every label is appended to
data/labels/human_labels.jsonl immediately; the latest entry per review wins, so going back and
relabelling is safe. Standard library only.
"""

import json
import random
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/snapshots/notion_2026-10-13"
SPEC = ROOT / "data/labels/D1_test_set.json"
OUT = ROOT / "data/labels/human_labels.jsonl"
GUIDE = ROOT / "research/labelling_guide.md"
SENTIMENTS = {"positive", "neutral", "negative"}


def load_items() -> list[dict[str, Any]]:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    wanted = set(spec["human_subset_ids"])
    processed = {}
    with (SNAPSHOT / "processed.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            if p["review_id"] in wanted:
                processed[p["review_id"]] = p
    items = [
        {
            "review_id": rid,
            "text": processed[rid]["text_model"],
            "translation": processed[rid]["text_en"] if processed[rid]["translated"] else None,
            "lang": processed[rid]["lang"],
        }
        for rid in sorted(wanted)
    ]
    random.Random(11).shuffle(items)
    return items


def load_labels() -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                labels[entry["review_id"]] = entry
    return labels


PAGE = (Path(__file__).with_suffix(".html")).read_text(encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            page = PAGE.replace("__PASS_MODE__", "true" if PASS_MODE else "false")
            self._send(200, page.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            state = {
                "items": ITEMS,
                "labels": load_labels(),
                "guide": GUIDE.read_text(encoding="utf-8"),
            }
            self._send(200, json.dumps(state).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/api/label":
            self._send(404, b"not found", "text/plain")
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["review_id"] in IDS and body["sentiment"] in SENTIMENTS
            entry = {
                "review_id": body["review_id"],
                "sentiment": body["sentiment"],
                "complaint": bool(body.get("complaint")),
                "unsure": bool(body.get("unsure")),
                "annotator": "human",
                "labelled_at": datetime.now(UTC).isoformat(),
            }
        except (ValueError, KeyError, AssertionError, TypeError):
            self._send(400, b"bad label", "text/plain")
            return
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        self._send(200, b"ok", "text/plain")

    def log_message(self, *args: Any) -> None:
        pass


if __name__ == "__main__":
    argv = sys.argv[1:]
    PASS_MODE = "--complaint-pass" in argv
    argv = [a for a in argv if a != "--complaint-pass"]
    ITEMS = load_items()
    if PASS_MODE:
        labelled = load_labels()
        ITEMS = [
            it
            for it in ITEMS
            if labelled.get(it["review_id"], {}).get("sentiment") == "negative"
            and not labelled[it["review_id"]].get("complaint")
        ]
    IDS = {it["review_id"] for it in ITEMS}
    port = int(argv[0]) if argv else 8765
    print(f"{len(ITEMS)} reviews, {len(load_labels())} already labelled -> http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
