"""Native local BGE-M3 provider.

Wraps `FlagEmbedding.BGEM3FlagModel` to produce dense vectors, sparse
lexical weights, and ColBERT multi-vectors in one pass, and exposes the
model's own scoring functions for each representation. This module only
works when the `local-bge-m3` extra is installed (see pyproject.toml);
importing it without that extra raises a clear, actionable error.

API calls here match the documented FlagEmbedding interface as of the
BGE-M3 reference documentation (bge-model.com): `encode()` returns a dict
with `dense_vecs`, `lexical_weights`, and `colbert_vecs`; scoring uses
`compute_lexical_matching_score()` and `colbert_score()` on the model
instance itself. If your installed FlagEmbedding version differs, verify
these method names with `python -c "from FlagEmbedding import
BGEM3FlagModel; help(BGEM3FlagModel)"` before relying on this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _import_bgem3_flag_model() -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise SystemExit(
            "Native BGE-M3 (dense + sparse + ColBERT) requires the local-bge-m3 "
            "extra. Install it with: uv sync --extra local-bge-m3"
        ) from exc
    return BGEM3FlagModel


@dataclass
class BGEM3Encoding:
    """Full BGE-M3 encoding output for a batch of texts."""

    dense_vecs: list[list[float]]
    lexical_weights: list[dict[str, float]]
    colbert_vecs: list[Any]


class BGEM3Provider:
    """Loads BGEM3FlagModel once and exposes encode/score helpers."""

    def __init__(self, model_name: str, use_fp16: bool = False) -> None:
        BGEM3FlagModel = _import_bgem3_flag_model()
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def encode(self, texts: list[str], max_length: int = 8192) -> BGEM3Encoding:
        """Encode texts requesting dense, sparse, and ColBERT representations."""
        output = self.model.encode(
            texts,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        dense_vecs = [vector.tolist() for vector in output["dense_vecs"]]
        colbert_vecs = list(output["colbert_vecs"])
        return BGEM3Encoding(
            dense_vecs=dense_vecs,
            lexical_weights=list(output["lexical_weights"]),
            colbert_vecs=colbert_vecs,
        )

    def score_dense(self, query_vec: list[float], doc_vec: list[float]) -> float:
        """BGE-M3 dense vectors are pre-normalized; inner product is cosine similarity."""
        return sum(q * d for q, d in zip(query_vec, doc_vec))

    def score_sparse(
        self, query_weights: dict[str, float], doc_weights: dict[str, float]
    ) -> float:
        return float(self.model.compute_lexical_matching_score(query_weights, doc_weights))

    def score_colbert(
        self, query_vecs: Any, doc_vecs: Any
    ) -> float:
        import numpy as np

        q = np.asarray(query_vecs) if not isinstance(query_vecs, np.ndarray) else query_vecs
        d = np.asarray(doc_vecs) if not isinstance(doc_vecs, np.ndarray) else doc_vecs
        return float(self.model.colbert_score(q, d).item())

    def score_hybrid(
        self,
        query_encoding: BGEM3Encoding,
        query_index: int,
        doc_encoding: BGEM3Encoding,
        doc_index: int,
        weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ) -> float:
        """Equal-weight combination of dense, sparse, and ColBERT scores.

        This follows BGE-M3's own documented hybrid-ranking formula
        (s_rank = w1*s_dense + w2*s_lex + w3*s_mul) with no additional
        normalization or fusion logic layered on top.
        """
        w_dense, w_lex, w_mul = weights
        s_dense = self.score_dense(
            query_encoding.dense_vecs[query_index], doc_encoding.dense_vecs[doc_index]
        )
        s_lex = self.score_sparse(
            query_encoding.lexical_weights[query_index],
            doc_encoding.lexical_weights[doc_index],
        )
        s_mul = self.score_colbert(
            query_encoding.colbert_vecs[query_index], doc_encoding.colbert_vecs[doc_index]
        )
        return w_dense * s_dense + w_lex * s_lex + w_mul * s_mul
