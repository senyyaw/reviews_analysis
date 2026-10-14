"""ASGI entrypoint for serverless hosting (Vercel detects `app` here).

Serves the committed, pre-analysed snapshots read-only: collection is disabled because a serverless
function has no persistent disk and a collection runs for minutes. Run the full API locally or in
Docker (README) to collect new reviews.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from review_analysis.api.app import create_app  # noqa: E402
from review_analysis.api.settings import Settings  # noqa: E402

app = create_app(Settings(snapshots_dir=ROOT / "data" / "snapshots", allow_collection=False))
