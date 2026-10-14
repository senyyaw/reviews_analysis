import json
from pathlib import Path

import pytest

from review_analysis.theme_classifier import ThemeClassifier

MODEL = Path(__file__).resolve().parents[1] / "models/themes"
present = (MODEL / "embedder.int8.onnx").exists() and (
    MODEL / "embedder.int8.onnx"
).stat().st_size > 1_000_000
needs_model = pytest.mark.skipif(not present, reason="theme model not present (git lfs pull)")


@needs_model
def test_obvious_sentences_get_the_right_theme():
    clf = ThemeClassifier(MODEL)
    cases = [
        ("The app keeps crashing every time I open a page.", "complaint", "c_stability"),
        (
            "They forced AI buttons everywhere and I cannot turn the AI off.",
            "complaint",
            "c_ai_intrusive",
        ),
        ("Typing on my iPad lags and the Apple Pencil does not work.", "complaint", "c_ipad"),
        ("I lost all my notes after the update, they are just gone.", "complaint", "c_data_loss"),
        (
            "It is way too expensive and the free plan block limit is ridiculous.",
            "complaint",
            "c_pricing",
        ),
        (
            "I use it for school, work and my personal life, it does everything.",
            "praise",
            "p_versatility",
        ),
    ]
    out = clf.predict([c[0] for c in cases], [c[1] for c in cases])  # type: ignore[misc]
    for (text, _, expected), themes in zip(cases, out, strict=True):
        assert expected in themes, (text, themes)


@needs_model
def test_kinds_are_kept_apart_and_metadata_is_complete():
    clf = ThemeClassifier(MODEL)
    out = clf.predict(["Love the templates!", "Crashes constantly."], ["praise", "complaint"])
    assert all(t.startswith("p_") for t in out[0]) and all(t.startswith("c_") for t in out[1])
    meta = json.loads((MODEL / "classifier.json").read_text())
    assert (
        meta["taxonomy_version"] and meta["cross_validation_on_int8"]["complaint"]["macro_f1"] > 0.5
    )
    assert len(meta["complaint"]["themes"]) == 17 and len(meta["praise"]["themes"]) == 8
