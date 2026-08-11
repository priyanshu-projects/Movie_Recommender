"""
src/evaluation/metrics.py

Ranking metrics for recommendation evaluation.

BERT4Rec primary metrics:
    NDCG@K   — quality of ranking (position-aware)
    Recall@K — how many relevant items appear in top-K
    Hit Rate@K — 1 if any relevant item is in top-K, else 0
    MRR@K    — reciprocal rank of first relevant item

SVD metrics:
    RMSE, MAE are computed by scikit-surprise directly.
    Precision@K and Recall@K use these functions below.
"""

import math


def ndcg_at_k(relevant: list, recommended: list, k: int | None = None) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    relevant:    list of ground-truth item IDs
    recommended: list of recommended item IDs (ranked)
    """
    if k is not None:
        recommended = recommended[:k]
    relevant_set = set(relevant)

    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommended)
        if item in relevant_set
    )
    ideal_hits = min(len(relevant_set), len(recommended))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(relevant: list, recommended: list, k: int | None = None) -> float:
    """Recall@K = |relevant ∩ top-K| / |relevant|."""
    if k is not None:
        recommended = recommended[:k]
    if not relevant:
        return 0.0
    hits = len(set(relevant) & set(recommended))
    return hits / len(relevant)


def precision_at_k(relevant: list, recommended: list, k: int | None = None) -> float:
    """Precision@K = |relevant ∩ top-K| / K."""
    if k is not None:
        recommended = recommended[:k]
    if not recommended:
        return 0.0
    hits = len(set(relevant) & set(recommended))
    return hits / len(recommended)


def hit_rate_at_k(target: int, recommended: list, k: int | None = None) -> float:
    """Hit Rate@K = 1.0 if target is in top-K, else 0.0."""
    if k is not None:
        recommended = recommended[:k]
    return 1.0 if target in recommended else 0.0


def mrr_at_k(relevant: list, recommended: list, k: int | None = None) -> float:
    """Mean Reciprocal Rank@K — reciprocal rank of first hit."""
    if k is not None:
        recommended = recommended[:k]
    relevant_set = set(relevant)
    for rank, item in enumerate(recommended):
        if item in relevant_set:
            return 1.0 / (rank + 1)
    return 0.0
