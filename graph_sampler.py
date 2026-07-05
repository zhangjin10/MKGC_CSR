"""
Graph sampling strategies
"""

import torch
import numpy as np
from collections import defaultdict
import random


class GraphSampler:
    """Sample subgraphs for training"""

    def __init__(self, triplets, num_entities, num_relations):
        self.triplets = triplets
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.build_adjacency()

    def build_adjacency(self):
        """Build adjacency lists"""
        self.adj_out = defaultdict(list)  # h -> [(r, t), ...]
        self.adj_in = defaultdict(list)   # t -> [(r, h), ...]

        for h, r, t in self.triplets:
            self.adj_out[h].append((r, t))
            self.adj_in[t].append((r, h))

    def sample_neighbors(self, entity, num_neighbors=10, direction='out'):
        """Sample neighbors of an entity"""
        if direction == 'out':
            neighbors = self.adj_out.get(entity, [])
        else:
            neighbors = self.adj_in.get(entity, [])

        if len(neighbors) <= num_neighbors:
            return neighbors
        else:
            return random.sample(neighbors, num_neighbors)

    def sample_subgraph(self, seed_entity, num_hops=2, max_nodes=100):
        """Sample a subgraph around a seed entity"""
        visited = set([seed_entity])
        frontier = [seed_entity]

        for hop in range(num_hops):
            new_frontier = []
            for entity in frontier:
                neighbors_out = self.sample_neighbors(entity, direction='out')
                neighbors_in = self.sample_neighbors(entity, direction='in')

                for r, neighbor in neighbors_out + neighbors_in:
                    if neighbor not in visited and len(visited) < max_nodes:
                        visited.add(neighbor)
                        new_frontier.append(neighbor)

            frontier = new_frontier
            if not frontier:
                break

        # Extract subgraph triplets
        subgraph_triplets = []
        for h, r, t in self.triplets:
            if h in visited and t in visited:
                subgraph_triplets.append((h, r, t))

        return subgraph_triplets, list(visited)


class RelationSampler:
    """Sample relations for training"""

    def __init__(self, triplets, num_relations):
        self.triplets = triplets
        self.num_relations = num_relations

        self.relation_triplets = defaultdict(list)
        for h, r, t in triplets:
            self.relation_triplets[r].append((h, r, t))

    def sample_by_relation(self, relation, num_samples=100):
        """Sample triplets with a specific relation"""
        triplets = self.relation_triplets.get(relation, [])

        if len(triplets) <= num_samples:
            return triplets
        else:
            return random.sample(triplets, num_samples)

    def balanced_relation_sampling(self, batch_size):
        """Sample triplets with balanced relation distribution"""
        samples_per_relation = batch_size // self.num_relations + 1

        sampled_triplets = []
        for r in range(self.num_relations):
            relation_samples = self.sample_by_relation(r, samples_per_relation)
            sampled_triplets.extend(relation_samples)

        random.shuffle(sampled_triplets)
        return sampled_triplets[:batch_size]

    def frequency_based_sampling(self, batch_size, temperature=1.0):
        """Sample relations based on frequency with temperature"""
        relation_counts = [len(self.relation_triplets[r]) for r in range(self.num_relations)]

        # Apply temperature
        probs = np.array(relation_counts, dtype=float)
        probs = probs ** (1.0 / temperature)
        probs = probs / probs.sum()

        # Sample relations
        sampled_relations = np.random.choice(
            self.num_relations,
            size=batch_size,
            p=probs,
            replace=True
        )

        # Sample triplets from selected relations
        sampled_triplets = []
        for r in sampled_relations:
            if len(self.relation_triplets[r]) > 0:
                triplet = random.choice(self.relation_triplets[r])
                sampled_triplets.append(triplet)

        return sampled_triplets


class StratifiedSampler:
    """Stratified sampling based on entity types"""

    def __init__(self, triplets, entity_types):
        self.triplets = triplets
        self.entity_types = entity_types

        self.type_triplets = defaultdict(list)
        for h, r, t in triplets:
            h_type = entity_types.get(h, 'unknown')
            t_type = entity_types.get(t, 'unknown')
            self.type_triplets[(h_type, t_type)].append((h, r, t))

    def sample_by_type_pair(self, head_type, tail_type, num_samples):
        """Sample triplets with specific entity type pair"""
        triplets = self.type_triplets.get((head_type, tail_type), [])

        if len(triplets) <= num_samples:
            return triplets
        else:
            return random.sample(triplets, num_samples)
