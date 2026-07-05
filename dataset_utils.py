"""
Dataset utilities and augmentation
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import random


class FilteredDataset(Dataset):
    """Dataset with filtered evaluation for link prediction"""

    def __init__(self, triplets, all_triplets, entity2id, relation2id):
        self.triplets = triplets
        self.all_triplets = set(all_triplets)
        self.entity2id = entity2id
        self.relation2id = relation2id

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        h, r, t = self.triplets[idx]
        return {
            'head': h,
            'relation': r,
            'tail': t,
            'all_triplets': self.all_triplets
        }

    def get_filter_mask(self, head, relation, num_entities):
        """Get mask for filtering existing triplets"""
        mask = torch.ones(num_entities)
        for t in range(num_entities):
            if (head, relation, t) in self.all_triplets:
                mask[t] = 0
        return mask


class DataAugmentation:
    """Data augmentation strategies for knowledge graphs"""

    def __init__(self, num_entities, num_relations):
        self.num_entities = num_entities
        self.num_relations = num_relations

    def inverse_relation_augmentation(self, triplets, relation_mapping):
        """Add inverse relations"""
        augmented = []
        for h, r, t in triplets:
            augmented.append((h, r, t))
            if r in relation_mapping:
                inv_r = relation_mapping[r]
                augmented.append((t, inv_r, h))
        return augmented

    def semantic_augmentation(self, triplets, entity_synonyms):
        """Replace entities with synonyms"""
        augmented = []
        for h, r, t in triplets:
            augmented.append((h, r, t))

            if h in entity_synonyms:
                for syn_h in entity_synonyms[h]:
                    augmented.append((syn_h, r, t))

            if t in entity_synonyms:
                for syn_t in entity_synonyms[t]:
                    augmented.append((h, r, syn_t))

        return augmented

    def random_mask_augmentation(self, features, mask_ratio=0.1):
        """Randomly mask features for robustness"""
        mask = torch.rand(features.shape) > mask_ratio
        masked_features = features * mask.float()
        return masked_features


class GraphStatistics:
    """Compute statistics for knowledge graphs"""

    def __init__(self, triplets):
        self.triplets = triplets
        self.head_per_relation = {}
        self.tail_per_relation = {}

    def compute_relation_cardinality(self):
        """Compute cardinality for each relation"""
        from collections import defaultdict

        head_per_rel = defaultdict(set)
        tail_per_rel = defaultdict(set)

        for h, r, t in self.triplets:
            head_per_rel[r].add(h)
            tail_per_rel[r].add(t)

        cardinality = {}
        for r in head_per_rel.keys():
            avg_heads = len(head_per_rel[r]) / max(1, len(tail_per_rel[r]))
            avg_tails = len(tail_per_rel[r]) / max(1, len(head_per_rel[r]))

            if avg_heads < 1.5 and avg_tails < 1.5:
                cardinality[r] = '1-to-1'
            elif avg_heads >= 1.5 and avg_tails < 1.5:
                cardinality[r] = 'N-to-1'
            elif avg_heads < 1.5 and avg_tails >= 1.5:
                cardinality[r] = '1-to-N'
            else:
                cardinality[r] = 'N-to-N'

        return cardinality

    def compute_entity_degree(self):
        """Compute in-degree and out-degree for entities"""
        from collections import defaultdict

        in_degree = defaultdict(int)
        out_degree = defaultdict(int)

        for h, r, t in self.triplets:
            out_degree[h] += 1
            in_degree[t] += 1

        return dict(in_degree), dict(out_degree)

    def compute_relation_frequency(self):
        """Compute frequency of each relation"""
        from collections import Counter

        relation_freq = Counter([r for h, r, t in self.triplets])
        return dict(relation_freq)
