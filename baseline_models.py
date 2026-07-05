"""
Baseline models for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransE(nn.Module):
    """TransE baseline model"""

    def __init__(self, num_entities, num_relations, embedding_dim=200):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, heads, relations, tails):
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        score = -torch.norm(h + r - t, p=2, dim=-1)
        return score


class DistMult(nn.Module):
    """DistMult baseline model"""

    def __init__(self, num_entities, num_relations, embedding_dim=200):
        super(DistMult, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, heads, relations, tails):
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        score = torch.sum(h * r * t, dim=-1)
        return score


class MultimodalFusion(nn.Module):
    """Multimodal fusion module"""

    def __init__(self, structural_dim, text_dim, visual_dim):
        super(MultimodalFusion, self).__init__()
        self.structural_dim = structural_dim
        self.text_dim = text_dim
        self.visual_dim = visual_dim

        self.text_proj = nn.Linear(text_dim, structural_dim)
        self.visual_proj = nn.Linear(visual_dim, structural_dim)
        self.fusion = nn.Linear(structural_dim * 3, structural_dim)

    def forward(self, structural, text, visual):
        text_proj = self.text_proj(text)
        visual_proj = self.visual_proj(visual)

        fused = torch.cat([structural, text_proj, visual_proj], dim=-1)
        output = self.fusion(fused)
        return output


class MKGformer(nn.Module):
    """MKGformer baseline with Transformer-based multimodal fusion"""

    def __init__(self, num_entities, num_relations, embedding_dim=200,
                 text_dim=768, visual_dim=768, num_heads=8):
        super(MKGformer, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.text_encoder = nn.Linear(text_dim, embedding_dim)
        self.visual_encoder = nn.Linear(visual_dim, embedding_dim)

        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads)
        self.fusion = MultimodalFusion(embedding_dim, embedding_dim, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, heads, relations, tails, head_text, head_visual, tail_text, tail_visual):
        h_struct = self.entity_embeddings(heads)
        t_struct = self.entity_embeddings(tails)
        r = self.relation_embeddings(relations)

        h_text = self.text_encoder(head_text)
        h_visual = self.visual_encoder(head_visual)
        t_text = self.text_encoder(tail_text)
        t_visual = self.visual_encoder(tail_visual)

        h_fused = self.fusion(h_struct, h_text, h_visual)
        t_fused = self.fusion(t_struct, t_text, t_visual)

        score = -torch.norm(h_fused + r - t_fused, p=2, dim=-1)
        return score


class LinkAwareFusion(nn.Module):
    """Link-aware fusion mechanism for LAFA"""

    def __init__(self, embedding_dim, num_modalities=3):
        super(LinkAwareFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * num_modalities, 128),
            nn.ReLU(),
            nn.Linear(128, num_modalities),
            nn.Softmax(dim=-1)
        )

    def forward(self, modality_features):
        concat = torch.cat(modality_features, dim=-1)
        weights = self.attention(concat)

        output = sum(w.unsqueeze(-1) * feat for w, feat in zip(weights.T, modality_features))
        return output


class LAFA(nn.Module):
    """LAFA baseline with link-aware fusion"""

    def __init__(self, num_entities, num_relations, embedding_dim=200,
                 text_dim=768, visual_dim=768):
        super(LAFA, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.text_encoder = nn.Linear(text_dim, embedding_dim)
        self.visual_encoder = nn.Linear(visual_dim, embedding_dim)

        self.link_aware_fusion = LinkAwareFusion(embedding_dim, num_modalities=3)

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, heads, relations, tails, head_text, head_visual, tail_text, tail_visual):
        h_struct = self.entity_embeddings(heads)
        t_struct = self.entity_embeddings(tails)
        r = self.relation_embeddings(relations)

        h_text = self.text_encoder(head_text)
        h_visual = self.visual_encoder(head_visual)
        t_text = self.text_encoder(tail_text)
        t_visual = self.visual_encoder(tail_visual)

        h_fused = self.link_aware_fusion([h_struct, h_text, h_visual])
        t_fused = self.link_aware_fusion([t_struct, t_text, t_visual])

        score = -torch.norm(h_fused + r - t_fused, p=2, dim=-1)
        return score
