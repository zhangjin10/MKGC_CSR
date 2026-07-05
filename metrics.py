"""
Evaluation metrics for link prediction
"""

import numpy as np
import torch


def compute_metrics(ranks):
    """
    Compute evaluation metrics from ranks

    Args:
        ranks: List or array of ranks

    Returns:
        Dictionary with MRR, Hits@1, Hits@3, Hits@10
    """
    ranks = np.array(ranks)

    mrr = np.mean(1.0 / ranks)
    hits_at_1 = np.mean(ranks <= 1)
    hits_at_3 = np.mean(ranks <= 3)
    hits_at_10 = np.mean(ranks <= 10)
    mean_rank = np.mean(ranks)

    return {
        'MRR': float(mrr),
        'Hits@1': float(hits_at_1),
        'Hits@3': float(hits_at_3),
        'Hits@10': float(hits_at_10),
        'MR': float(mean_rank)
    }


def rank_evaluation(scores, true_index, filter_indices=None):
    """
    Compute rank of true entity given scores

    Args:
        scores: Scores for all entities
        true_index: Index of true entity
        filter_indices: Indices to filter out (other true triplets)

    Returns:
        Filtered rank
    """
    if filter_indices is not None:
        scores = scores.copy()
        scores[filter_indices] = -1e10

    sorted_indices = np.argsort(-scores)
    rank = np.where(sorted_indices == true_index)[0][0] + 1

    return rank


def compute_reciprocal_rank(rank):
    """Compute reciprocal rank"""
    return 1.0 / rank


def compute_hits_at_k(rank, k):
    """Check if rank is within top-k"""
    return 1.0 if rank <= k else 0.0


class MetricsCalculator:
    """Calculate various metrics for evaluation"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated metrics"""
        self.ranks = []
        self.reciprocal_ranks = []

    def add_rank(self, rank):
        """Add a rank to accumulate"""
        self.ranks.append(rank)
        self.reciprocal_ranks.append(1.0 / rank)

    def get_metrics(self):
        """Get accumulated metrics"""
        if len(self.ranks) == 0:
            return None

        return compute_metrics(self.ranks)

    def get_mean_rank(self):
        """Get mean rank"""
        return np.mean(self.ranks) if self.ranks else 0.0

    def get_mrr(self):
        """Get mean reciprocal rank"""
        return np.mean(self.reciprocal_ranks) if self.reciprocal_ranks else 0.0


def evaluate_by_relation_type(ranks_by_type):
    """
    Evaluate metrics by relation type

    Args:
        ranks_by_type: Dictionary mapping relation types to lists of ranks

    Returns:
        Dictionary with metrics for each relation type
    """
    results = {}

    for rel_type, ranks in ranks_by_type.items():
        if len(ranks) > 0:
            results[rel_type] = compute_metrics(ranks)
        else:
            results[rel_type] = None

    return results


def compute_ndcg(scores, relevant_indices, k=10):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG)

    Args:
        scores: Predicted scores
        relevant_indices: Indices of relevant items
        k: Cutoff position

    Returns:
        NDCG@k score
    """
    sorted_indices = np.argsort(-scores)[:k]

    dcg = 0.0
    for i, idx in enumerate(sorted_indices):
        if idx in relevant_indices:
            dcg += 1.0 / np.log2(i + 2)

    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_indices), k)))

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def compute_map(scores_list, relevant_indices_list):
    """
    Compute Mean Average Precision (MAP)

    Args:
        scores_list: List of score arrays
        relevant_indices_list: List of relevant indices for each query

    Returns:
        MAP score
    """
    average_precisions = []

    for scores, relevant_indices in zip(scores_list, relevant_indices_list):
        sorted_indices = np.argsort(-scores)

        precisions = []
        num_relevant = 0

        for i, idx in enumerate(sorted_indices):
            if idx in relevant_indices:
                num_relevant += 1
                precision = num_relevant / (i + 1)
                precisions.append(precision)

        if precisions:
            average_precisions.append(np.mean(precisions))

    return np.mean(average_precisions) if average_precisions else 0.0
