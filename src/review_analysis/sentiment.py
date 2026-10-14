"""Sentiment + complaint inference with the fine-tuned int8 ONNX models (no torch at runtime).

Two models, routed by language (see research/results/finetune.md):
- models/sentiment-en:    fine-tuned cardiffnlp RoBERTa, for "en" and "und" (short or undetermined);
- models/sentiment-multi: fine-tuned clapAI XLM-R, for every other language.
On the human-labelled test set the router scores macro-F1 0.784 and accuracy 0.89. That is
significantly better than the multilingual model alone, and the English model alone fails on
non-English text.

Each model outputs 4 logits: softmax over [negative, neutral, positive], plus a sigmoid complaint
score. Decision rule (DESIGN.md §5.4): negative if P(neg) >= 0.30, otherwise the argmax of
neutral/positive. Complaint if P(complaint) >= 0.5.
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

Label = Literal["negative", "neutral", "positive"]
CLASSES: tuple[Label, Label, Label] = ("negative", "neutral", "positive")
NEG_THRESHOLD = 0.30
COMPLAINT_THRESHOLD = 0.5
ENGLISH_LANGS = frozenset({"en", "und"})
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def models_dir() -> Path:
    return Path(os.environ.get("REVIEW_ANALYSIS_MODELS_DIR", DEFAULT_MODELS_DIR))


@dataclass(frozen=True)
class SentimentResult:
    label: Label
    p_negative: float
    p_neutral: float
    p_positive: float
    p_complaint: float
    complaint: bool
    model: str


def decide(p_neg: float, p_neu: float, p_pos: float) -> Label:
    if p_neg >= NEG_THRESHOLD:
        return "negative"
    return "neutral" if p_neu >= p_pos else "positive"


class OnnxSentimentModel:
    """One exported model directory: model.int8.onnx, tokenizer.json and model.json.

    batch_size defaults to 1 on purpose. Dynamic int8 quantisation computes activation scales per
    batch, so in batches of 32 a review's label could depend on which other reviews shared its
    batch (6 of 1,143 labels changed). One review at a time is deterministic, and on CPU it is
    also faster (no padding): 8.2 s vs 9.6 s for the pool.
    """

    def __init__(self, directory: Path, *, batch_size: int = 1, threads: int | None = None) -> None:
        self.directory = directory
        self.batch_size = batch_size
        self.threads = threads
        self.meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        self.name = str(self.meta["name"])

    @cached_property
    def _session(self) -> ort.InferenceSession:
        options = ort.SessionOptions()
        if self.threads:
            options.intra_op_num_threads = self.threads
        return ort.InferenceSession(
            str(self.directory / "model.int8.onnx"), options, providers=["CPUExecutionProvider"]
        )

    @cached_property
    def _tokenizer(self) -> Tokenizer:
        tok = Tokenizer.from_file(str(self.directory / "tokenizer.json"))
        tok.enable_truncation(int(self.meta["max_len"]))
        pad = str(self.meta["pad_token"])
        tok.enable_padding(pad_id=tok.token_to_id(pad), pad_token=pad)
        return tok

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """(n, 4) array: P(negative), P(neutral), P(positive), P(complaint)."""
        if not texts:
            return np.zeros((0, 4))
        # Sort by length so each batch pads to a similar size, then restore the order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), 4))
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self._tokenizer.encode_batch([texts[i] for i in idx])
            feeds = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            }
            logits = self._session.run(None, feeds)[0]
            sent = np.exp(logits[:, :3] - logits[:, :3].max(axis=1, keepdims=True))
            sent /= sent.sum(axis=1, keepdims=True)
            out[idx] = np.concatenate([sent, 1 / (1 + np.exp(-logits[:, 3:]))], axis=1)
        return out


class SentimentRouter:
    def __init__(self, directory: Path | None = None) -> None:
        root = directory or models_dir()
        self.english = OnnxSentimentModel(root / "sentiment-en")
        self.multilingual = OnnxSentimentModel(root / "sentiment-multi")

    def model_for(self, lang: str) -> OnnxSentimentModel:
        return self.english if lang in ENGLISH_LANGS else self.multilingual

    def predict(self, texts: Sequence[str], langs: Sequence[str]) -> list[SentimentResult]:
        if len(texts) != len(langs):
            raise ValueError("texts and langs must have the same length")
        results: list[SentimentResult | None] = [None] * len(texts)
        for model in (self.english, self.multilingual):
            idx = [i for i, lang in enumerate(langs) if self.model_for(lang) is model]
            if not idx:
                continue
            for i, p in zip(idx, model.predict_proba([texts[i] for i in idx]), strict=True):
                p_neg, p_neu, p_pos, p_comp = (float(x) for x in p)
                results[i] = SentimentResult(
                    label=decide(p_neg, p_neu, p_pos),
                    p_negative=p_neg,
                    p_neutral=p_neu,
                    p_positive=p_pos,
                    p_complaint=p_comp,
                    complaint=p_comp >= COMPLAINT_THRESHOLD,
                    model=model.name,
                )
        return [r for r in results if r is not None]
