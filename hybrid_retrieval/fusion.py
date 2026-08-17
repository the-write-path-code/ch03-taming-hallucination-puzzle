"""Standalone score and rank fusion algorithms for hybrid retrieval.

Implements Reciprocal Rank Fusion (RRF) and Relative Score Fusion (RSF).
Both functions are standalone, deterministic, typed, and unit-testable without
external dependencies or model downloads.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_k: int = 60,
) -> dict[str, float]:
    """Combine multiple ranked document lists using Reciprocal Rank Fusion (RRF).

    For a document at zero-based position `rank` in a list, its contribution is:
        1.0 / (rrf_k + rank + 1)

    Args:
        ranked_lists: A list of ranked document ID lists (ordered from best to worst).
        rrf_k: Ranking constant parameter. Must be non-negative (>= 0).

    Returns:
        A mapping of doc_id to aggregated RRF score. Callers sort by (-score, doc_id).

    Raises:
        ValueError: If rrf_k < 0.
    """
    if rrf_k < 0:
        raise ValueError(f"rrf_k must be non-negative, got {rrf_k}.")

    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()
        rank = 0
        for doc_id in ranked_list:
            if doc_id in seen_in_list:
                continue
            seen_in_list.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            rank += 1
    return scores


def relative_score_fusion(
    score_maps: list[dict[str, float]],
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Combine multiple score maps using min-max normalization and weighted sum.

    For each score map independently, scores are normalized to [0, 1]:
        normalized = (score - min_score) / (max_score - min_score)

    If all values in a score map are identical (max == min), every document present
    in that map receives a normalized score of 1.0 to avoid division by zero.
    A document missing from a score map contributes 0.0 for that source.

    Args:
        score_maps: A non-empty list of {doc_id: raw_score} mappings.
        weights: Optional list of non-negative weights matching len(score_maps).
                 If omitted, equal weights are used. Weights are normalized by sum.

    Returns:
        A mapping of doc_id to weighted relative fused score.

    Raises:
        ValueError: If score_maps is empty, weights length doesn't match, any weight
                    is negative, or total weight sum is zero.
    """
    if not score_maps:
        raise ValueError("score_maps must not be empty.")

    num_sources = len(score_maps)
    if weights is None:
        norm_weights = [1.0 / num_sources] * num_sources
    else:
        if len(weights) != num_sources:
            raise ValueError(
                f"Length of weights ({len(weights)}) must match score_maps ({num_sources})."
            )
        if any(w < 0.0 for w in weights):
            raise ValueError(f"Weights must be non-negative, got {weights}.")
        total_weight = sum(weights)
        if total_weight <= 0.0:
            raise ValueError(f"Total weight sum must be positive, got {total_weight}.")
        norm_weights = [w / total_weight for w in weights]

    all_doc_ids: set[str] = set()
    for s_map in score_maps:
        all_doc_ids.update(s_map.keys())

    fused_scores: dict[str, float] = {doc_id: 0.0 for doc_id in all_doc_ids}

    for score_map, weight in zip(score_maps, norm_weights):
        if not score_map:
            continue
        values = list(score_map.values())
        min_score = min(values)
        max_score = max(values)
        spread = max_score - min_score

        for doc_id in all_doc_ids:
            if doc_id in score_map:
                if spread == 0.0:
                    norm_val = 1.0
                else:
                    norm_val = (score_map[doc_id] - min_score) / spread
            else:
                norm_val = 0.0
            fused_scores[doc_id] += weight * norm_val

    return fused_scores
