"""Sentiment scoring — a model-agnostic scorer abstraction.

Two implementations:

* :class:`FakeScorer` — a deterministic, dependency-free lexicon scorer. This is the
  **default test path**: it never downloads weights and gives stable outputs, so the
  whole pipeline (parse -> score -> store -> summarise) is exercisable in pytest with
  no model files and no network.
* :class:`FinbertScorer` — optional, **local-only** real FinBERT (``ProsusAI/finbert``
  via ``transformers``/``torch``, both already in the project deps). It loads weights
  with ``local_files_only=True`` and **never** downloads them silently and **never**
  calls a hosted inference API. If the weights are absent it raises with explicit
  instructions instead of fetching anything.

Output is the model-agnostic :class:`~options_system.sentiment.schema.SentimentScore`
(``positive``/``negative``/``neutral`` probabilities, the derived
``sentiment_score = positive - negative``, the model name and a version/hash).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from options_system.sentiment.schema import SentimentScore


@runtime_checkable
class Scorer(Protocol):
    """Anything that turns a piece of text into a :class:`SentimentScore`."""

    name: str

    def score_text(self, text: str) -> SentimentScore: ...


def _now() -> datetime:
    return datetime.now(UTC)


# --- deterministic fake scorer (test + offline default) --------------------- #

# Small finance-tinted lexicons. Deterministic and transparent — this is NOT a model,
# only a stand-in so the plumbing is testable without weights.
_POSITIVE_WORDS = frozenset(
    {
        "beat",
        "beats",
        "surge",
        "surges",
        "record",
        "growth",
        "gain",
        "gains",
        "strong",
        "rally",
        "upgrade",
        "boom",
        "optimism",
        "profit",
        "rebound",
    }
)
_NEGATIVE_WORDS = frozenset(
    {
        "miss",
        "misses",
        "recession",
        "cut",
        "cuts",
        "fall",
        "falls",
        "weak",
        "selloff",
        "downgrade",
        "fear",
        "fears",
        "loss",
        "slump",
        "crash",
        "default",
    }
)


class FakeScorer:
    """Deterministic lexicon scorer. No weights, no network, stable outputs."""

    name = "fake-lexicon"
    version = "v1"

    def __init__(self, now: datetime | None = None) -> None:
        # Fixed clock injectable so tests can assert on the full struct deterministically.
        self._fixed_now = now

    def _scored_at(self) -> datetime:
        return self._fixed_now if self._fixed_now is not None else _now()

    def score_text(self, text: str) -> SentimentScore:
        tokens = [t.strip(".,!?:;\"'()[]").lower() for t in text.split()]
        pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
        # Map raw counts to a 3-way distribution. Neutral mass shrinks as signal grows.
        base_neutral = 1.0
        weights = [pos + 0.0, neg + 0.0, base_neutral]
        total = sum(weights) or 1.0
        p, n, neu = (w / total for w in weights)
        return SentimentScore(
            positive_score=p,
            negative_score=n,
            neutral_score=neu,
            sentiment_score=p - n,
            model_name=f"{self.name}-{self.version}",
            model_version_or_hash=self.version,
            scored_at=self._scored_at(),
        )


# --- optional local FinBERT ------------------------------------------------- #


class FinbertWeightsMissing(RuntimeError):
    """Raised when local FinBERT weights are not present (we never download them)."""


class FinbertScorer:
    """Local FinBERT scorer. Loads weights with ``local_files_only=True`` only.

    Construction is cheap; the model is loaded lazily on the first
    :meth:`score_text`. If the weights are not on disk, loading raises
    :class:`FinbertWeightsMissing` with instructions — it never reaches out to the
    network and never calls a hosted inference endpoint.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str | None = None,
        version_hash: str | None = None,
    ) -> None:
        self.name = model_name
        self.device = device
        # The resolved local snapshot revision (git commit hash) when known — stamped
        # as model_version_or_hash on every score so a re-score with different weights
        # is distinguishable. Falls back to the model path when not provided.
        self.version_hash = version_hash
        self._pipeline: bool | None = None
        self._id2label: dict[int, str] | None = None
        # Heavy objects, populated lazily by _ensure_loaded (kept Any to avoid
        # importing torch/transformers types at module load).
        self._torch: Any = None
        self._tok: Any = None
        self._model: Any = None

    @classmethod
    def available(cls, model_name: str = "ProsusAI/finbert") -> bool:
        """True iff the weights/config are cached locally (no download attempted)."""
        try:
            from transformers import AutoConfig

            AutoConfig.from_pretrained(model_name, local_files_only=True)
            return True
        except Exception:  # noqa: BLE001 - any failure means "not locally available"
            return False

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except Exception as exc:  # noqa: BLE001 - deps missing
            raise FinbertWeightsMissing(
                f"transformers/torch unavailable ({exc}); cannot run local FinBERT."
            ) from exc
        try:
            cfg = AutoConfig.from_pretrained(self.name, local_files_only=True)
            tok = AutoTokenizer.from_pretrained(self.name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.name, local_files_only=True
            )
        except Exception as exc:  # noqa: BLE001 - weights not cached locally
            raise FinbertWeightsMissing(
                f"Local FinBERT weights for {self.name!r} were not found and will NOT be "
                f"downloaded automatically. To enable real scoring, fetch the weights once, "
                f"deliberately, into the local HF cache (e.g. "
                f"`huggingface-cli download {self.name}`), then re-run. ({exc})"
            ) from exc
        model.eval()
        if self.device:
            model.to(self.device)
        self._torch = torch
        self._tok = tok
        self._model = model
        self._id2label = {int(k): v.lower() for k, v in cfg.id2label.items()}
        self._pipeline = True  # sentinel: loaded

    def _score_from_probs(self, probs: list[float], scored_at: datetime) -> SentimentScore:
        assert self._id2label is not None
        by_label = {self._id2label[i]: float(probs[i]) for i in range(len(probs))}
        p = by_label.get("positive", 0.0)
        n = by_label.get("negative", 0.0)
        neu = by_label.get("neutral", 0.0)
        return SentimentScore(
            positive_score=p,
            negative_score=n,
            neutral_score=neu,
            sentiment_score=p - n,
            model_name=self.name,
            model_version_or_hash=(
                self.version_hash or getattr(self._model.config, "_name_or_path", None)
            ),
            scored_at=scored_at,
        )

    def score_text(self, text: str) -> SentimentScore:
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str], *, batch_size: int = 64) -> list[SentimentScore]:
        """Score many texts in fixed-size batches (deterministic output order).

        Inference only: ``model.eval()`` was set at load; ``torch.inference_mode()``
        disables autograd; padding/truncation keep every (short) title well under
        FinBERT's 512-token limit. Runs on ``self.device`` when set (e.g. ``cuda``).
        """
        if not texts:
            return []
        self._ensure_loaded()
        torch = self._torch
        scored_at = _now()
        out: list[SentimentScore] = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                inputs = self._tok(
                    chunk, return_tensors="pt", truncation=True, max_length=512, padding=True
                )
                if self.device:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().tolist()
                out.extend(self._score_from_probs(p, scored_at) for p in probs)
        return out
