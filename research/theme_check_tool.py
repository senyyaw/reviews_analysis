"""Human spot-check of the LLM theme labels (sentence level).

    uv run python research/theme_check_tool.py        # then open http://127.0.0.1:8767

30 sentences, stratified: about 2 per theme plus a few the LLM left untagged, drawn with a fixed
seed. Each is shown with its review and the theme(s) the LLM assigned. Keys: 1 = correct,
2 = partly (one theme right, or one missing), 3 = wrong. Answers go to
data/labels/theme_spotcheck.jsonl and give the LLM's theme precision as judged by a human.
"""

import json
import random
import sys
from collections import defaultdict
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/labels/themes_qwen.jsonl"
INPUT = ROOT / "data/labels/sentences_for_llm.jsonl"
OUT = ROOT / "data/labels/theme_spotcheck.jsonl"
VERDICTS = {"1": "correct", "2": "partly", "3": "wrong"}


def build_items(n_per_theme: int = 2, n_untagged: int = 4, seed: int = 5) -> list[dict[str, Any]]:
    tax = json.loads((ROOT / "data/themes.json").read_text())
    names = {t["id"]: t["name"] for k in ("complaint", "praise") for t in tax[k]}
    inp = {json.loads(line)["sid"]: json.loads(line) for line in INPUT.read_text().splitlines()}
    lab = {
        json.loads(line)["sid"]: json.loads(line).get("themes", [])
        for line in LABELS.read_text().splitlines()
    }
    by_theme: dict[str, list[str]] = defaultdict(list)
    for sid, themes in sorted(lab.items()):
        for t in themes:
            by_theme[t].append(sid)
    rng = random.Random(seed)
    chosen: list[str] = []
    for t in sorted(by_theme):
        pool = [s for s in by_theme[t] if s not in chosen]
        chosen += rng.sample(pool, min(n_per_theme, len(pool)))
    untagged = sorted(s for s, t in lab.items() if not t)
    chosen += rng.sample(untagged, min(n_untagged, len(untagged)))
    rng.shuffle(chosen)
    chosen = chosen[:30] if len(chosen) > 30 else chosen
    return [
        {
            "sid": s,
            "kind": inp[s]["kind"],
            "text": inp[s]["text"],
            "review": inp[s]["review"][:600],
            "themes": [names.get(t, t) for t in lab[s]] or ["(no theme)"],
        }
        for s in chosen
    ]


PAGE = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            done = {}
            if OUT.exists():
                for line in OUT.read_text().splitlines():
                    row = json.loads(line)
                    done[row["sid"]] = row["verdict"]
            self._send(200, json.dumps({"items": ITEMS, "done": done}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            item = next(it for it in ITEMS if it["sid"] == body["sid"])
            row = {
                "sid": item["sid"],
                "verdict": VERDICTS[body["verdict"]],
                "llm_themes": item["themes"],
                "checked_at": datetime.now(UTC).isoformat(),
            }
        except (StopIteration, KeyError, ValueError, TypeError):
            self._send(400, b"bad answer", "text/plain")
            return
        with OUT.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        self._send(200, b"ok", "text/plain")

    def log_message(self, *args: Any) -> None:
        pass


if __name__ == "__main__":
    ITEMS = build_items()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8767
    print(f"{len(ITEMS)} sentences -> http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
