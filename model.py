"""
MKGC-CSR: Multimodal Knowledge Graph Completion with Causal Semantic Reasoning
Main model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import numpy as np


class RotatEEmbedding(nn.Module):
    """RotatE structural embedding"""
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RotatEEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim * 2)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

        nn.init.uniform_(self.entity_embedding.weight, -1.0, 1.0)
        nn.init.uniform_(self.relation_embedding.weight, -1.0, 1.0)

    def forward(self, entities):
        return self.entity_embedding(entities)

    def get_relation_embedding(self, relations):
        return self.relation_embedding(relations)


class VisualConceptDictionary(nn.Module):
    """Visual Concept Dictionary using K-Means clustering"""
    def __init__(self, visual_dim, num_concepts=64):
        super(VisualConceptDictionary, self).__init__()
        self.num_concepts = num_concepts
        self.visual_dim = visual_dim

        # Initialize concept centroids
        self.centroids = nn.Parameter(torch.randn(num_concepts, visual_dim))
        nn.init.xavier_uniform_(self.centroids)

        # Store empirical priors (will be updated during training)
        self.register_buffer('priors', torch.ones(num_concepts) / num_concepts)

    def update_from_kmeans(self, visual_features, labels, counts):
        """Update centroids and priors from K-Means clustering results"""
        with torch.no_grad():
            for k in range(self.num_concepts):
                mask = (labels == k)
                if mask.sum() > 0:
                    self.centroids[k] = visual_features[mask].mean(dim=0)

            # Update empirical priors
            total_count = counts.sum()
            self.priors = counts.float() / total_count

    def get_centroids(self):
        return self.centroids

    def get_priors(self):
        return self.priors


class StratifiedCausalAttention(nn.Module):
    """Stratified Causal Attention for backdoor adjustment"""
    def __init__(self, structural_dim, text_dim, visual_dim, num_concepts):
        super(StratifiedCausalAttention, self).__init__()
        self.num_concepts = num_concepts
        self.query_dim = structural_dim * 2 + text_dim

        # MLP for computing attention over visual concepts
        self.mlp = nn.Sequential(
            nn.Linear(self.query_dim + visual_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, visual_dim)
        )

        self.layer_norm = LayerNorm(visual_dim)

    def forward(self, query, visual_raw, visual_dictionary, priors):
        """
        Args:
            query: [batch_size, query_dim] - fused structural and textual features
            visual_raw: [batch_size, visual_dim] - raw visual features
            visual_dictionary: [num_concepts, visual_dim] - visual concept centroids
            priors: [num_concepts] - empirical priors for each concept
        Returns:
            deconfounded_visual: [batch_size, visual_dim]
        """
        batch_size = query.size(0)
        num_concepts = visual_dictionary.size(0)

        # Expand dimensions for broadcasting
        query_expanded = query.unsqueeze(1).expand(-1, num_concepts, -1)  # [B, K, query_dim]
        visual_raw_expanded = visual_raw.unsqueeze(1).expand(-1, num_concepts, -1)  # [B, K, visual_dim]
        visual_dict_expanded = visual_dictionary.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, visual_dim]

        # Concatenate query, concept, and raw visual
        mlp_input = torch.cat([query_expanded, visual_dict_expanded, visual_raw_expanded], dim=-1)  # [B, K, query_dim + 2*visual_dim]

        # Apply MLP to get representation for each concept
        concept_representations = self.mlp(mlp_input)  # [B, K, visual_dim]

        # Weight by empirical priors
        priors_expanded = priors.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, concept_representations.size(-1))  # [B, K, visual_dim]
        weighted_representations = concept_representations * priors_expanded  # [B, K, visual_dim]

        # Aggregate over all concepts (backdoor adjustment)
        aggregated = weighted_representations.sum(dim=1)  # [B, visual_dim]

        # Residual connection and layer normalization
        deconfounded_visual = self.layer_norm(visual_raw + aggregated)

        return deconfounded_visual


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining structural and multimodal features"""
    def __init__(self, structural_dim, multimodal_dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(structural_dim * 2 + multimodal_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, structural_features, multimodal_features):
        """
        Args:
            structural_features: [batch_size, structural_dim * 2]
            multimodal_features: [batch_size, multimodal_dim]
        Returns:
            fused_features: [batch_size, structural_dim * 2]
        """
        concat_features = torch.cat([structural_features, multimodal_features], dim=-1)
        gate_weight = self.gate(concat_features)

        # Ensure multimodal_features matches structural_features dimension
        if multimodal_features.size(-1) != structural_features.size(-1):
            multimodal_features = F.linear(multimodal_features,
                                          torch.randn(structural_features.size(-1), multimodal_features.size(-1)).to(multimodal_features.device))

        fused = gate_weight * structural_features + (1 - gate_weight) * multimodal_features
        return fused


class MKGC_CSR(nn.Module):
    """
    Complete MKGC-CSR model for multimodal knowledge graph completion
    """
    def __init__(self, num_entities, num_relations,
                 structural_dim=200, text_dim=768, visual_dim=768,
                 num_visual_concepts=64):
        super(MKGC_CSR, self).__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.structural_dim = structural_dim
        self.text_dim = text_dim
        self.visual_dim = visual_dim

        # Structural encoder (RotatE)
        self.structural_encoder = RotatEEmbedding(num_entities, num_relations, structural_dim)

        # Text encoder projection (assuming BERT features are pre-extracted)
        self.text_projection = nn.Linear(text_dim, text_dim)

        # Visual encoder projection (assuming ViT features are pre-extracted)
        self.visual_projection = nn.Linear(visual_dim, visual_dim)

        # Visual concept dictionary
        self.visual_dictionary = VisualConceptDictionary(visual_dim, num_visual_concepts)

        # Stratified causal attention
        self.causal_attention = StratifiedCausalAttention(
            structural_dim, text_dim, visual_dim, num_visual_concepts
        )

        # Project deconfounded visual + text to match structural dimension
        self.multimodal_projection = nn.Linear(text_dim + visual_dim, structural_dim * 2)

        # Gated fusion
        self.gated_fusion = GatedFusion(structural_dim, structural_dim * 2)

        self.dropout = nn.Dropout(0.1)

    def forward(self, head_entities, relations, tail_entities,
                head_text_features, head_visual_features,
                tail_text_features, tail_visual_features):
        """
        Args:
            head_entities: [batch_size] - head entity indices
            relations: [batch_size] - relation indices
            tail_entities: [batch_size] - tail entity indices
            head_text_features: [batch_size, text_dim] - pre-extracted BERT features
            head_visual_features: [batch_size, visual_dim] - pre-extracted ViT features
            tail_text_features: [batch_size, text_dim]
            tail_visual_features: [batch_size, visual_dim]
        Returns:
            score: [batch_size] - triplet plausibility scores
        """
        # Structural embeddings
        head_structural = self.structural_encoder(head_entities)  # [B, structural_dim * 2]
        tail_structural = self.structural_encoder(tail_entities)  # [B, structural_dim * 2]
        relation_embedding = self.structural_encoder.get_relation_embedding(relations)  # [B, structural_dim]

        # Process head entity
        head_text = self.text_projection(head_text_features)
        head_visual_raw = self.visual_projection(head_visual_features)

        # Query for causal attention (structural + textual)
        head_query = torch.cat([head_structural, head_text], dim=-1)

        # Causal deconfounding for head visual features
        head_visual_deconf = self.causal_attention(
            head_query,
            head_visual_raw,
            self.visual_dictionary.get_centroids(),
            self.visual_dictionary.get_priors()
        )

        # Fuse text and deconfounded visual
        head_multimodal = torch.cat([head_text, head_visual_deconf], dim=-1)
        head_multimodal = self.multimodal_projection(head_multimodal)
        head_multimodal = self.dropout(head_multimodal)

        # Gated fusion for head
        head_final = self.gated_fusion(head_structural, head_multimodal)

        # Process tail entity (same procedure)
        tail_text = self.text_projection(tail_text_features)
        tail_visual_raw = self.visual_projection(tail_visual_features)
        tail_query = torch.cat([tail_structural, tail_text], dim=-1)

        tail_visual_deconf = self.causal_attention(
            tail_query,
            tail_visual_raw,
            self.visual_dictionary.get_centroids(),
            self.visual_dictionary.get_priors()
        )

        tail_multimodal = torch.cat([tail_text, tail_visual_deconf], dim=-1)
        tail_multimodal = self.multimodal_projection(tail_multimodal)
        tail_multimodal = self.dropout(tail_multimodal)

        tail_final = self.gated_fusion(tail_structural, tail_multimodal)

        # RotatE scoring function
        score = self.rotate_score(head_final, relation_embedding, tail_final)

        return score

    def rotate_score(self, head, relation, tail):
        """
        RotatE scoring: score = -||h ∘ r - t||
        where ∘ is complex multiplication
        """
        # Split into real and imaginary parts
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        # Relation as phase (convert to complex rotation)
        phase_relation = relation / (self.structural_dim / np.pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        # Complex multiplication: h ∘ r
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        # Distance: ||h ∘ r - t||
        re_diff = re_score - re_tail
        im_diff = im_score - im_tail

        score = torch.stack([re_diff, im_diff], dim=0).norm(dim=0).sum(dim=-1)

        return -score  # Negative distance as score (higher is better)

    def predict(self, head_entities, relations, head_text_features, head_visual_features,
                all_tail_embeddings=None):
        """
        Prediction mode for link prediction
        Returns scores for all possible tail entities
        """
        batch_size = head_entities.size(0)

        # Get head representations
        head_structural = self.structural_encoder(head_entities)
        relation_embedding = self.structural_encoder.get_relation_embedding(relations)

        head_text = self.text_projection(head_text_features)
        head_visual_raw = self.visual_projection(head_visual_features)
        head_query = torch.cat([head_structural, head_text], dim=-1)

        head_visual_deconf = self.causal_attention(
            head_query,
            head_visual_raw,
            self.visual_dictionary.get_centroids(),
            self.visual_dictionary.get_priors()
        )

        head_multimodal = torch.cat([head_text, head_visual_deconf], dim=-1)
        head_multimodal = self.multimodal_projection(head_multimodal)
        head_final = self.gated_fusion(head_structural, head_multimodal)

        # Compute scores for all entities
        if all_tail_embeddings is None:
            all_tail_embeddings = self.structural_encoder.entity_embedding.weight

        # Broadcast and compute scores
        scores = []
        for i in range(batch_size):
            h = head_final[i:i+1].expand(self.num_entities, -1)
            r = relation_embedding[i:i+1].expand(self.num_entities, -1)
            score = self.rotate_score(h, r, all_tail_embeddings)
            scores.append(score)

        return torch.stack(scores, dim=0)
