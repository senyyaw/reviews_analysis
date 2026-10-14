"""Serve-time theme tagging, without an LLM (research/theme_classifier.py, export_theme_model.py).

multilingual-e5-base sentence embeddings (int8 ONNX, mean pooling + L2 normalisation inside the
graph), then one-vs-rest logistic regression per theme with tuned thresholds. It is a student of
the Qwen3.8-27B teacher labels. Cross-validated against the teacher (GroupKFold by review): theme
prevalence off by 3.3 pp (complaints) and 1.1 pp (praise) on average.

Sentences are embedded one at a time, as for the sentiment models: dynamic int8 quantisation
computes activation scales per batch, so batching would make results depend on neighbours.
"""

import json
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from review_analysis.sentiment import models_dir

Kind = Literal["complaint", "praise"]


class ThemeClassifier:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or models_dir() / "themes"
        self.meta = json.loads((self.directory / "classifier.json").read_text())
        self.name = f"classifier:{self.meta['embedder'].split('/')[-1]}"
        self._heads = {
            kind: (
                self.meta[kind]["themes"],
                np.array(self.meta[kind]["coef"], dtype=np.float32),
                np.array(self.meta[kind]["intercept"], dtype=np.float32),
                np.array(self.meta[kind]["thresholds"], dtype=np.float32),
            )
            for kind in ("complaint", "praise")
        }

    @cached_property
    def _session(self) -> ort.InferenceSession:
        return ort.InferenceSession(
            str(self.directory / "embedder.int8.onnx"), providers=["CPUExecutionProvider"]
        )

    @cached_property
    def _tokenizer(self) -> Tokenizer:
        tok = Tokenizer.from_file(str(self.directory / "tokenizer.json"))
        tok.enable_truncation(int(self.meta["max_len"]))
        return tok

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 768), dtype=np.float32)
        prefix = self.meta["prefix"]
        for i, text in enumerate(texts):
            enc = self._tokenizer.encode(prefix + text)
            feeds = {
                "input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            }
            out[i] = self._session.run(None, feeds)[0][0]
        return out

    def predict(self, texts: list[str], kinds: list[Kind]) -> list[list[str]]:
        """Theme ids per sentence; complaint sentences get complaint themes, praise praise."""
        if len(texts) != len(kinds):
            raise ValueError("texts and kinds must have the same length")
        result: list[list[str]] = [[] for _ in texts]
        emb = self.embed(texts)
        for kind, (ids, coef, intercept, thr) in self._heads.items():
            rows = [i for i, k in enumerate(kinds) if k == kind]
            if not rows:
                continue
            probs = 1 / (1 + np.exp(-(emb[rows] @ coef.T + intercept)))
            for r, p in zip(rows, probs, strict=True):
                result[r] = [t for t, on in zip(ids, p >= thr, strict=True) if on]
        return result
