"""Static insight charts (PNG) for the README and the sample report.

Style follows the dataviz reference palette:
- light surface #fcfcfb, primary/secondary/muted ink, hairline solid gridlines;
- blue/orange categorical slots for A vs B, and blue (praise) vs red (complaint) for polarity.
  Both pairs were checked with the palette validator for colour-vision-deficiency separation.
- thin marks, 2px lines, 8px markers with a surface ring, 10%-opacity CI bands;
- selective direct labels; text in ink, never in series colour.
Each title states the finding, computed from the data, and each chart shows n and uncertainty.

Requires the optional extra:  uv sync --extra report
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f0f6fe", "#86b6ef", "#2a78d6", "#0d366b"])
DPI = 200


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _clean(ax: Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)


HEADER_IN = 0.75  # inches reserved above the plots for the title and subtitle
LEGEND_IN = 0.30  # plus a legend row when the chart has one


def _save(
    fig: Any, path: Path, title: str, subtitle: str, *, legend: list[Any] | None = None
) -> None:
    """A fixed-size header strip (title, subtitle, optional legend row) that never overlaps."""
    h = fig.get_figheight()
    header = HEADER_IN + (LEGEND_IN if legend else 0)
    fig.text(
        0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK
    )
    fig.text(0.01, 1 - 0.42 / h, subtitle, ha="left", va="top", fontsize=9, color=INK2)
    if legend:
        fig.legend(
            handles=legend,
            loc="upper left",
            bbox_to_anchor=(0.005, 1 - 0.66 / h),
            ncol=len(legend),
            fontsize=9,
            handletextpad=0.4,
            columnspacing=1.6,
        )
    fig.tight_layout(rect=(0, 0, 1, 1 - header / h))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _key(color: str, label: str, marker: str = "s") -> Any:
    return Line2D([], [], marker=marker, linestyle="", color=color, markersize=8, label=label)


def star_distribution(data: dict[str, Any], path: Path) -> None:
    _style()
    a, b = data["star_distribution"]["A"], data["star_distribution"]["B"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    stars = [str(s) for s in range(1, 6)]
    width = 0.36
    for offset, (name, series, color) in enumerate(
        (("A: random 300, last 12 months", a, BLUE), ("B: 100 most recent", b, ORANGE))
    ):
        xs = [i + (offset - 0.5) * (width + 0.04) for i in range(5)]
        shares = [series[s]["share"] * 100 for s in stars]
        lo = [shares[i] - series[s]["ci_low"] * 100 for i, s in enumerate(stars)]
        hi = [series[s]["ci_high"] * 100 - shares[i] for i, s in enumerate(stars)]
        ax.bar(xs, shares, width=width, color=color, label=name, zorder=2)
        ax.errorbar(
            xs,
            shares,
            yerr=[lo, hi],
            fmt="none",
            ecolor=INK2,
            elinewidth=0.9,
            capsize=2.5,
            zorder=3,
        )
        for i in (0, 4):  # label the extremes only
            ax.text(
                xs[i],
                shares[i] + hi[i] + 1.2,
                f"{shares[i]:.0f}%",
                ha="center",
                fontsize=9,
                color=INK2,
            )
    ax.set_xticks(range(5), [f"{s}★" for s in stars])
    ax.set_ylabel("share of reviews (%)")
    _clean(ax)
    one_a, one_b = a["1"]["share"] * 100, b["1"]["share"] * 100
    _save(
        fig,
        path,
        f"Recent reviews are harsher: {one_b:.0f}% one-star vs {one_a:.0f}% over the year",
        "Share of written reviews per star rating · error bars = 95% Wilson CI · "
        f"A n={data['samples']['A']['n']}, B n={data['samples']['B']['n']}",
        legend=[_key(BLUE, "A: random 300, last 12 months"), _key(ORANGE, "B: 100 most recent")],
    )


def monthly(data: dict[str, Any], path: Path) -> None:
    _style()
    rows = data["monthly"]
    months = [r["month"] for r in rows]
    x = list(range(len(rows)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"hspace": 0.3})
    mean = [r["mean_rating"] for r in rows]
    ax1.fill_between(
        x,
        [r["mean_ci"][0] for r in rows],
        [r["mean_ci"][1] for r in rows],
        color=BLUE,
        alpha=0.10,
        linewidth=0,
    )
    ax1.plot(
        x,
        mean,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        solid_capstyle="round",
    )
    ax1.set_ylabel("mean stars")
    ax1.set_ylim(1, 5)
    peak = max(x, key=lambda i: mean[i])
    for i in (peak, x[-1]):
        ax1.annotate(
            f"{mean[i]:.2f}★",
            (x[i], mean[i]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
            color=INK2,
        )
    neg = [r["negative"]["share"] * 100 for r in rows]
    ax2.fill_between(
        x,
        [r["negative"]["ci_low"] * 100 for r in rows],
        [r["negative"]["ci_high"] * 100 for r in rows],
        color=BLUE,
        alpha=0.10,
        linewidth=0,
    )
    ax2.plot(
        x,
        neg,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    ax2.set_ylabel("1–2★ share (%)")
    ax2.set_ylim(0, 100)
    for i in (min(x, key=lambda i: neg[i]), x[-1]):
        ax2.annotate(
            f"{neg[i]:.0f}%",
            (x[i], neg[i]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
            color=INK2,
        )
    for ax in (ax1, ax2):
        _clean(ax)
    ax2.set_xticks(
        x, [f"{m[2:4]}-{m[5:]}\nn={r['n']}" for m, r in zip(months, rows, strict=True)], fontsize=8
    )
    worst_drop = min(range(1, len(x)), key=lambda i: mean[i] - mean[i - 1])
    _save(
        fig,
        path,
        f"Ratings peaked in {months[peak]} ({mean[peak]:.2f}★) and fell to {mean[-1]:.2f}★ by "
        f"{months[-1]}",
        f"All written reviews, last 12 months (n={data['n_pool']}) · shaded = 95% CI · biggest "
        f"month-on-month drop: {months[worst_drop - 1]} → {months[worst_drop]}",
    )


def stars_vs_text(data: dict[str, Any], path: Path) -> None:
    _style()
    counts = data["stars_vs_text"]["counts"]
    comp = data["stars_vs_text"]["complaint_share"]
    stars = [str(s) for s in range(5, 0, -1)]
    labels = ["negative", "neutral", "positive"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4), gridspec_kw={"width_ratios": [1.15, 1]})
    mat = [[counts[s][lab] / max(1, sum(counts[s].values())) for lab in labels] for s in stars]
    ax1.imshow(mat, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            ax1.text(
                j,
                i,
                f"{v:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if v > 0.55 else INK,
            )
    ax1.set_xticks(range(3), ["text negative", "text neutral", "text positive"])
    ax1.set_yticks(range(5), [f"{s}★  (n={sum(counts[s].values())})" for s in stars])
    ax1.grid(False)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.tick_params(length=0)
    ax1.set_title("What the text says, per star rating", fontsize=10, color=INK2)
    shares = [comp[s]["share"] * 100 for s in stars]
    lo = [shares[i] - comp[s]["ci_low"] * 100 for i, s in enumerate(stars)]
    hi = [comp[s]["ci_high"] * 100 - shares[i] for i, s in enumerate(stars)]
    ax2.barh(range(5), shares, height=0.5, color=RED, zorder=2)
    ax2.errorbar(
        shares,
        range(5),
        xerr=[lo, hi],
        fmt="none",
        ecolor=INK2,
        elinewidth=0.9,
        capsize=2.5,
        zorder=3,
    )
    for i, v in enumerate(shares):
        ax2.text(min(v + hi[i] + 2, 101), i, f"{v:.0f}%", va="center", fontsize=9, color=INK2)
    ax2.set_yticks(range(5), [f"{s}★" for s in stars])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 112)
    ax2.set_xlabel("reviews containing a complaint (%)")
    ax2.grid(axis="y", visible=False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(length=0)
    ax2.set_title("Complaint detected (complaint head)", fontsize=10, color=INK2)
    four = comp["4"]["share"] * 100
    _save(
        fig,
        path,
        f"{four:.0f}% of 4★ reviews still contain a complaint: the cheapest stars to win",
        f"Fine-tuned sentiment + complaint model on all {data['n_pool']} reviews · "
        "rows sum to 100% · error bars = 95% Wilson CI",
    )


def _dumbbell(
    ax: Axes,
    terms: list[dict[str, Any]],
    color_a: str,
    color_b: str,
    title: str,
    key_a: str,
    key_b: str,
) -> None:
    ys = list(range(len(terms)))
    for y, t in zip(ys, terms, strict=True):
        a, b = t[key_a] * 100, t[key_b] * 100
        ax.plot([a, b], [y, y], color=AXIS, linewidth=2, zorder=1, solid_capstyle="round")
        ax.scatter([b], [y], s=55, color=color_b, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter([a], [y], s=55, color=color_a, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_yticks(ys, [t["term"] for t in terms])
    ax.invert_yaxis()
    ax.set_xlabel("% of reviews mentioning it")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=10, color=INK2)


def distinctive_terms(data: dict[str, Any], path: Path) -> None:
    _style()
    d = data["distinctive_terms"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.4))
    _dumbbell(
        ax1, d["negative"], RED, BLUE, "Over-represented in negative reviews", "share_a", "share_b"
    )
    _dumbbell(
        ax2,
        d["positive"],
        RED,
        BLUE,
        "Over-represented in neutral/positive reviews",
        "share_a",
        "share_b",
    )
    top = d["negative"][0]
    _save(
        fig,
        path,
        "Unhappy reviews talk about stability and devices; happy ones about organising life",
        f"Fightin' Words (top {len(d['negative'])} per side, |z| > 1.96), prior from "
        f"{d['n_background']:,} reviews of 14 other productivity apps · e.g. “{top['term']}”: "
        f"{top['share_a']:.0%} vs {top['share_b']:.0%} of reviews",
        legend=[
            _key(RED, f"negative reviews (n={d['n_a']})", "o"),
            _key(BLUE, f"neutral + positive reviews (n={d['n_b']})", "o"),
        ],
    )


def drivers(data: dict[str, Any], path: Path) -> None:
    _style()
    d = data["drivers"]
    low, high = d["towards_low"][:10], d["towards_high"][:10]
    rows = low + list(reversed(high))
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ys = list(range(len(rows)))
    for y, r in zip(ys, rows, strict=True):
        color = RED if r["odds_ratio"] > 1 else BLUE
        ax.barh(y, r["odds_ratio"] - 1, left=1, height=0.55, color=color, zorder=2)
        tip = r["odds_ratio"]
        ax.text(
            tip * (1.04 if tip > 1 else 0.96),
            y,
            f"×{tip:.2f}  (n={r['df']})",
            va="center",
            ha="left" if tip > 1 else "right",
            fontsize=8.5,
            color=INK2,
        )
    ax.axvline(1, color=AXIS, linewidth=1)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())  # no "3x10^-1" minor labels
    ax.set_xlim(0.25, 5)
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ["×0.25", "×0.5", "×1", "×2", "×4"])
    ax.set_yticks(ys, [r["term"] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("odds of a 1–2★ rating when the review mentions it (log scale)")
    top = low[0]
    _save(
        fig,
        path,
        f"Mentioning “{top['term']}” multiplies the odds of a 1–2★ rating by "
        f"{top['odds_ratio']:.1f}",
        f"Driver analysis: logistic regression 1–2★ (n={d['n_low']}) vs 4–5★ (n={d['n_high']}), "
        f"cross-validated AUC {d['cv_auc']:.2f}; drivers stable in ≥90% of 200 bootstraps",
    )


def most_mentioned(data: dict[str, Any], path: Path) -> None:
    _style()
    d = data["most_mentioned"]
    rows = d["terms"]
    n = d["n_reviews"]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ys = list(range(len(rows)))
    neg = [r["negative"] / n * 100 for r in rows]
    rest = [r["not_negative"] / n * 100 for r in rows]
    gap = 0.15  # surface gap between the two segments, in percentage points
    ax.barh(ys, neg, height=0.6, color=RED, zorder=2)
    ax.barh(
        ys,
        [max(0.0, v - gap) for v in rest],
        left=[v + gap for v in neg],
        height=0.6,
        color=BLUE,
        zorder=2,
    )
    for y, r in zip(ys, rows, strict=True):
        share = r["share"] * 100
        neg_pct = r["negative"] / max(1, r["count"]) * 100
        ax.text(
            share + 0.4,
            y,
            f"{share:.0f}%  ·  {neg_pct:.0f}% negative",
            va="center",
            fontsize=8.5,
            color=INK2,
        )
    ax.set_yticks(ys, [r["term"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(r["share"] for r in rows) * 100 * 1.45)
    ax.set_xlabel("% of reviews mentioning it")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    top = rows[0]
    worst = max(rows[:10], key=lambda r: r["negative"] / max(1, r["count"]))
    _save(
        fig,
        path,
        f"Most talked about: “{top['term']}” ({top['share']:.0%} of reviews); "
        f"most negative of the top 10: “{worst['term']}”",
        f"Product aspects ranked by share of the {n} reviews that mention them (generic "
        "sentiment words excluded) · bar split by the review's tone",
        legend=[_key(RED, "in negative reviews"), _key(BLUE, "in neutral + positive reviews")],
    )


ALL = {
    "most_mentioned": most_mentioned,
    "star_distribution": star_distribution,
    "monthly_trend": monthly,
    "stars_vs_text": stars_vs_text,
    "distinctive_terms": distinctive_terms,
    "drivers": drivers,
}


def render_all(data: dict[str, Any], out_dir: Path) -> list[Path]:
    paths = []
    for name, fn in ALL.items():
        path = out_dir / f"{name}.png"
        fn(data, path)
        paths.append(path)
    return paths
