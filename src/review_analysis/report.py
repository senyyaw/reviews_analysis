"""HTML report for one collection: the /metrics, /insights and comparison data, with charts.

The page is self-contained apart from Chart.js (pinned, from cdnjs). Every number comes from the API
payloads; review text is HTML-escaped by Jinja2 autoescaping and never injected as markup. Served at
GET /v1/collections/{id}/report and exported to static HTML by `review-analysis export-site`.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Undefined, select_autoescape

from review_analysis import __version__

TEMPLATES = Path(__file__).resolve().parent / "templates"
CHART_JS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"

_env = Environment(
    loader=FileSystemLoader(TEMPLATES),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _missing(v: Any) -> bool:
    return v is None or isinstance(v, Undefined)


_env.filters["pct"] = lambda v, d=0: "–" if _missing(v) else f"{v * 100:.{d}f}%"
_env.filters["stars"] = lambda v, d=2: "–" if _missing(v) else f"{v:.{d}f}★"
_env.filters["signed"] = lambda v, d=1: "–" if _missing(v) else f"{v:+.{d}f}".replace("-", "−")


def _by_star(dist: dict[Any, Any]) -> list[dict[str, Any]]:
    return [{"star": s, **dist[str(s)]} for s in range(1, 6) if str(s) in dist]


def chart_data(
    metrics: dict[str, Any], insights: dict[str, Any], comparison: dict[str, Any] | None
) -> dict[str, Any]:
    """The compact data the page's charts draw (embedded as JSON)."""
    m = metrics["metrics"]
    dist = {str(k): v for k, v in m["distribution"].items()}
    svt = metrics.get("stars_vs_text") or {}
    themes = insights.get("themes") or {}
    keep = ("id", "name", "prevalence", "prevalence_ci", "rating_impact", "n_reviews", "priority")
    terms = insights["distinctive_terms"]
    drivers = insights["drivers"]
    return {
        "stars": _by_star(dist),
        "stars_vs_text": [
            {"star": s, **svt["counts"][str(s)], "complaint": svt["complaint_share"][str(s)]}
            for s in range(1, 6)
        ]
        if svt
        else [],
        "monthly": [
            {"month": g["key"], "n": g["n"], "mean": g["mean_rating"], **g["negative"]}
            for g in m["by_month"]
        ],
        "complaint_themes": [
            {k: t.get(k) for k in keep} for t in themes.get("complaint", []) if t["n_reviews"]
        ],
        "praise_themes": [
            {k: t.get(k) for k in keep} for t in themes.get("praise", []) if t["n_reviews"]
        ],
        "most_mentioned": insights["most_mentioned"]["terms"][:15],
        "distinctive": {
            "negative": terms.get("negative", [])[:10],
            "positive": terms.get("positive", [])[:10],
        }
        if terms.get("available")
        else None,
        "drivers": {
            "low": drivers.get("towards_low", [])[:10],
            "high": drivers.get("towards_high", [])[:10],
        }
        if drivers.get("available")
        else None,
        "comparison": [
            {
                "name": t["name"],
                "kind": t["kind"],
                "base": t["base_share"],
                "target": t["target_share"],
                "status": t["status"],
            }
            for t in (comparison or {}).get("themes", [])
            if t["kind"] == "complaint"
        ],
    }


def render_report(
    summary: dict[str, Any],
    metrics: dict[str, Any],
    insights: dict[str, Any],
    comparison: dict[str, Any] | None = None,
    *,
    links: dict[str, str] | None = None,
) -> str:
    """links overrides the API's download links (a static export points them at static files)."""
    return _env.get_template("report.html").render(
        summary=summary,
        m=metrics["metrics"],
        metrics=metrics,
        insights=insights,
        comparison=comparison,
        data=chart_data(metrics, insights, comparison),
        links=links or summary["links"],
        chart_js=CHART_JS,
        version=__version__,
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )


def render_index(entries: Sequence[tuple[dict[str, Any], ...]], api_url: str = "") -> str:
    """Landing page of the static site: one row per collection, linking to its report."""
    return _env.get_template("index.html").render(
        entries=entries,
        api_url=api_url,
        version=__version__,
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )
