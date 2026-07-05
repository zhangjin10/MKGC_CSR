"""
Evaluation metrics and robustness testing for MKGC-CSR
"""

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class MetricsEvaluator:
    """Compute evaluation metrics for link prediction"""

    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.all_triplets = preprocessor.get_all_triplets()

    def compute_filtered_rank(self, head, relation, tail_true, scores):
        """
        Compute filtered rank for a triplet

        Args:
            head: Head entity ID
            relation: Relation ID
            tail_true: True tail entity ID
            scores: Scores for all candidate tails [num_entities]

        Returns:
            rank: Filtered rank of true tail
        """
        # Filter out other true triplets
        for t_other in range(len(scores)):
            if t_other != tail_true and (head, relation, t_other) in self.all_triplets:
                scores[t_other] = -1e10

        # Get rank
        sorted_indices = np.argsort(-scores)
        rank = np.where(sorted_indices == tail_true)[0][0] + 1
        return rank

    def evaluate_triplets(self, triplets, corrupt_mode='tail'):
        """
        Evaluate a list of triplets

        Args:
            triplets: List of (head, relation, tail) triplets
            corrupt_mode: 'tail' or 'head' - which entity to predict

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_ranks = []

        with torch.no_grad():
            for h, r, t in tqdm(triplets, desc=f"Evaluating ({corrupt_mode})"):
                if corrupt_mode == 'tail':
                    # Predict tail given head and relation
                    h_tensor = torch.tensor([h], device=self.device)
                    r_tensor = torch.tensor([r], device=self.device)

                    h_text = self.preprocessor.text_features.get(h, torch.zeros(768)).unsqueeze(0).to(self.device)
                    h_visual = self.preprocessor.visual_features.get(h, torch.zeros(768)).unsqueeze(0).to(self.device)

                    scores = []
                    for t_candidate in range(self.model.num_entities):
                        t_tensor = torch.tensor([t_candidate], device=self.device)
                        t_text = self.preprocessor.text_features.get(t_candidate, torch.zeros(768)).unsqueeze(0).to(self.device)
                        t_visual = self.preprocessor.visual_features.get(t_candidate, torch.zeros(768)).unsqueeze(0).to(self.device)

                        score = self.model(h_tensor, r_tensor, t_tensor, h_text, h_visual, t_text, t_visual)
                        scores.append(score.item())

                    scores = np.array(scores)
                    rank = self.compute_filtered_rank(h, r, t, scores)
                    all_ranks.append(rank)

                else:  # corrupt_mode == 'head'
                    # Predict head given tail and relation
                    r_tensor = torch.tensor([r], device=self.device)
                    t_tensor = torch.tensor([t], device=self.device)

                    t_text = self.preprocessor.text_features.get(t, torch.zeros(768)).unsqueeze(0).to(self.device)
                    t_visual = self.preprocessor.visual_features.get(t, torch.zeros(768)).unsqueeze(0).to(self.device)

                    scores = []
                    for h_candidate in range(self.model.num_entities):
                        h_tensor = torch.tensor([h_candidate], device=self.device)
                        h_text = self.preprocessor.text_features.get(h_candidate, torch.zeros(768)).unsqueeze(0).to(self.device)
                        h_visual = self.preprocessor.visual_features.get(h_candidate, torch.zeros(768)).unsqueeze(0).to(self.device)

                        score = self.model(h_tensor, r_tensor, t_tensor, h_text, h_visual, t_text, t_visual)
                        scores.append(score.item())

                    scores = np.array(scores)
                    rank = self.compute_filtered_rank(t, r, h, scores)  # Note: reverse for head prediction
                    all_ranks.append(rank)

        # Compute metrics
        all_ranks = np.array(all_ranks)
        mrr = np.mean(1.0 / all_ranks)
        hits_at_1 = np.mean(all_ranks <= 1)
        hits_at_3 = np.mean(all_ranks <= 3)
        hits_at_10 = np.mean(all_ranks <= 10)

        return {
            'MRR': mrr,
            'Hits@1': hits_at_1,
            'Hits@3': hits_at_3,
            'Hits@10': hits_at_10,
            'mean_rank': np.mean(all_ranks)
        }

    def evaluate_by_relation_type(self, triplets):
        """
        Evaluate performance by relation type (1-to-1, 1-to-N, N-to-1, N-to-N)
        """
        # Compute relation cardinalities
        head_per_relation = defaultdict(set)
        tail_per_relation = defaultdict(set)

        for h, r, t in self.preprocessor.get_all_triplets():
            head_per_relation[r].add(h)
            tail_per_relation[r].add(t)

        # Classify relations
        relation_types = {}
        for r in range(len(self.preprocessor.relation2id)):
            avg_heads = len(head_per_relation[r]) / max(1, len(tail_per_relation[r]))
            avg_tails = len(tail_per_relation[r]) / max(1, len(head_per_relation[r]))

            if avg_heads < 1.5 and avg_tails < 1.5:
                relation_types[r] = '1-to-1'
            elif avg_heads >= 1.5 and avg_tails < 1.5:
                relation_types[r] = 'N-to-1'
            elif avg_heads < 1.5 and avg_tails >= 1.5:
                relation_types[r] = '1-to-N'
            else:
                relation_types[r] = 'N-to-N'

        # Group triplets by relation type
        triplets_by_type = defaultdict(list)
        for h, r, t in triplets:
            rel_type = relation_types.get(r, 'N-to-N')
            triplets_by_type[rel_type].append((h, r, t))

        # Evaluate each type
        results = {}
        for rel_type in ['1-to-1', '1-to-N', 'N-to-1', 'N-to-N']:
            if rel_type in triplets_by_type and len(triplets_by_type[rel_type]) > 0:
                print(f"\nEvaluating {rel_type} relations ({len(triplets_by_type[rel_type])} triplets)...")

                # Head prediction
                head_metrics = self.evaluate_triplets(triplets_by_type[rel_type], corrupt_mode='head')

                # Tail prediction
                tail_metrics = self.evaluate_triplets(triplets_by_type[rel_type], corrupt_mode='tail')

                results[rel_type] = {
                    'head_prediction': head_metrics,
                    'tail_prediction': tail_metrics
                }

        return results


class RobustnessEvaluator:
    """Evaluate model robustness under noisy conditions"""

    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device

    def add_gaussian_noise(self, features, noise_std=0.1):
        """Add Gaussian noise to features"""
        noise = torch.randn_like(features) * noise_std
        return features + noise

    def evaluate_with_noise(self, triplets, noise_std=0.1):
        """
        Evaluate model with Gaussian noise added to visual features

        Args:
            triplets: List of (head, relation, tail) triplets
            noise_std: Standard deviation of Gaussian noise

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_ranks = []
        all_triplets_set = self.preprocessor.get_all_triplets()

        print(f"Evaluating with Gaussian noise (σ={noise_std})...")

        with torch.no_grad():
            for h, r, t in tqdm(triplets):
                h_tensor = torch.tensor([h], device=self.device)
                r_tensor = torch.tensor([r], device=self.device)

                h_text = self.preprocessor.text_features.get(h, torch.zeros(768)).unsqueeze(0).to(self.device)
                h_visual = self.preprocessor.visual_features.get(h, torch.zeros(768)).unsqueeze(0).to(self.device)

                # Add noise to visual features
                h_visual_noisy = self.add_gaussian_noise(h_visual, noise_std)

                scores = []
                for t_candidate in range(self.model.num_entities):
                    t_tensor = torch.tensor([t_candidate], device=self.device)
                    t_text = self.preprocessor.text_features.get(t_candidate, torch.zeros(768)).unsqueeze(0).to(self.device)
                    t_visual = self.preprocessor.visual_features.get(t_candidate, torch.zeros(768)).unsqueeze(0).to(self.device)

                    # Add noise to tail visual features
                    t_visual_noisy = self.add_gaussian_noise(t_visual, noise_std)

                    score = self.model(h_tensor, r_tensor, t_tensor, h_text, h_visual_noisy, t_text, t_visual_noisy)
                    scores.append(score.item())

                scores = np.array(scores)

                # Filtered ranking
                for t_other in range(len(scores)):
                    if t_other != t and (h, r, t_other) in all_triplets_set:
                        scores[t_other] = -1e10

                sorted_indices = np.argsort(-scores)
                rank = np.where(sorted_indices == t)[0][0] + 1
                all_ranks.append(rank)

        # Compute metrics
        all_ranks = np.array(all_ranks)
        mrr = np.mean(1.0 / all_ranks)
        hits_at_1 = np.mean(all_ranks <= 1)
        hits_at_3 = np.mean(all_ranks <= 3)
        hits_at_10 = np.mean(all_ranks <= 10)

        return {
            'MRR': mrr,
            'Hits@1': hits_at_1,
            'Hits@3': hits_at_3,
            'Hits@10': hits_at_10,
            'noise_std': noise_std
        }

    def robustness_analysis(self, triplets, noise_levels=[0.0, 0.1, 0.2, 0.3]):
        """
        Run robustness analysis across different noise levels

        Args:
            triplets: List of (head, relation, tail) triplets
            noise_levels: List of noise standard deviations to test

        Returns:
            Dictionary mapping noise levels to metrics
        """
        results = {}

        for noise_std in noise_levels:
            metrics = self.evaluate_with_noise(triplets, noise_std)
            results[noise_std] = metrics

            print(f"\nNoise σ={noise_std}: MRR={metrics['MRR']:.4f}, "
                  f"Hits@1={metrics['Hits@1']:.4f}, "
                  f"Hits@3={metrics['Hits@3']:.4f}, "
                  f"Hits@10={metrics['Hits@10']:.4f}")

        return results


def compute_mrr_degradation(clean_mrr, noisy_mrr):
    """Compute relative MRR degradation"""
    return (clean_mrr - noisy_mrr) / clean_mrr * 100


def print_comparison_table(baseline_results, mkgc_csr_results, noise_levels):
    """Print comparison table for robustness analysis"""
    print("\n" + "="*80)
    print("Robustness Comparison Table")
    print("="*80)
    print(f"{'Noise σ':<10} {'Baseline MRR':<15} {'MKGC-CSR MRR':<15} {'Baseline Deg.%':<15} {'MKGC-CSR Deg.%':<15}")
    print("-"*80)

    baseline_clean = baseline_results[0.0]['MRR']
    mkgc_clean = mkgc_csr_results[0.0]['MRR']

    for noise in noise_levels:
        baseline_mrr = baseline_results[noise]['MRR']
        mkgc_mrr = mkgc_csr_results[noise]['MRR']

        baseline_deg = compute_mrr_degradation(baseline_clean, baseline_mrr)
        mkgc_deg = compute_mrr_degradation(mkgc_clean, mkgc_mrr)

        print(f"{noise:<10.1f} {baseline_mrr:<15.4f} {mkgc_mrr:<15.4f} {baseline_deg:<15.2f} {mkgc_deg:<15.2f}")

    print("="*80)
