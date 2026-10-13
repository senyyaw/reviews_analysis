"""Offline machine translation of non-English reviews into English, for keyword extraction only.

Model: facebook/m2m100 (MIT licence, 100 languages, runs offline). NLLB is avoided because of its
CC-BY-NC licence. Translation is done SENTENCE BY SENTENCE: M2M100 is trained on sentence pairs and,
given several sentences at once, often drops the first one (on our data it dropped review titles
such as "Algo desepcionante"). Repetition is blocked during generation, because the model otherwise
sometimes loops ("I don't know how to use it, I don't know how to use it, ...").

Results are cached per review_id in the snapshot (translations.json), so the deployed API and later
runs never need the model. Sentiment and themes do NOT use translations;
they run on the original text with multilingual models.

Requires the optional extra:  uv sync --extra translate
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

MODEL_ID = "facebook/m2m100_1.2B"


class Translation(BaseModel):
    src_lang: str
    text_en: str
    model: str


class TranslationCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, Translation] = {}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.entries = {k: Translation.model_validate(v) for k, v in raw.items()}

    def get(self, review_id: str) -> Translation | None:
        return self.entries.get(review_id)

    def put(self, review_id: str, translation: Translation) -> None:
        self.entries[review_id] = translation

    def save(self) -> None:
        data = {k: v.model_dump() for k, v in sorted(self.entries.items())}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


class M2MTranslator:
    """Thin wrapper around the Hugging Face M2M100 model. Imported lazily: torch is optional."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        self.model_id = model_id
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.tokenizer: Any = M2M100Tokenizer.from_pretrained(model_id)
        model: Any = M2M100ForConditionalGeneration.from_pretrained(model_id)
        self.model: Any = model.to(self.device)
        self.model.eval()
        self.supported = set(self.tokenizer.lang_code_to_id)

    def translate(self, texts: Sequence[str], src_lang: str) -> list[str]:
        """Translate each text by splitting it into sentences and translating those in one batch."""
        from review_analysis.preprocessing.sentences import split_sentences

        per_text = [split_sentences(t) or [t] for t in texts]
        flat = [s for sents in per_text for s in sents]
        translated = iter(self._translate_sentences(flat, src_lang))
        return [" ".join(next(translated) for _ in sents).strip() for sents in per_text]

    def _translate_sentences(self, sentences: Sequence[str], src_lang: str) -> list[str]:
        import torch

        self.tokenizer.src_lang = src_lang
        out: list[str] = []
        for start in range(0, len(sentences), 16):
            chunk = list(sentences[start : start + 16])
            batch = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **batch,
                    forced_bos_token_id=self.tokenizer.get_lang_id("en"),
                    num_beams=4,
                    no_repeat_ngram_size=4,
                    max_new_tokens=min(256, 3 * batch["input_ids"].shape[1] + 16),
                )
            out.extend(
                str(t) for t in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return out
