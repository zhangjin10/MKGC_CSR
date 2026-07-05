"""
Attention mechanisms for multimodal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention"""

    def __init__(self, temperature=1.0):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.temperature

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        output, attention_weights = self.attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        output = self.dropout(output)

        return output, attention_weights


class CrossModalAttention(nn.Module):
    """Cross-modal attention between different modalities"""

    def __init__(self, query_dim, key_dim, hidden_dim):
        super(CrossModalAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)

        self.attention = ScaledDotProductAttention(temperature=hidden_dim ** 0.5)

    def forward(self, query, key_value):
        Q = self.query_proj(query).unsqueeze(1)
        K = self.key_proj(key_value).unsqueeze(1)
        V = self.value_proj(key_value).unsqueeze(1)

        output, attention_weights = self.attention(Q, K, V)
        output = output.squeeze(1)

        return output, attention_weights


class CoAttention(nn.Module):
    """Co-attention mechanism for joint reasoning"""

    def __init__(self, dim1, dim2, hidden_dim):
        super(CoAttention, self).__init__()
        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)

        self.attention1 = nn.Linear(hidden_dim, 1)
        self.attention2 = nn.Linear(hidden_dim, 1)

    def forward(self, features1, features2):
        proj1 = self.proj1(features1)
        proj2 = self.proj2(features2)

        # Compute attention scores
        scores1 = self.attention1(torch.tanh(proj1 + proj2))
        scores2 = self.attention2(torch.tanh(proj1 + proj2))

        weights1 = F.softmax(scores1, dim=0)
        weights2 = F.softmax(scores2, dim=0)

        attended1 = (weights1 * features1).sum(dim=0)
        attended2 = (weights2 * features2).sum(dim=0)

        return attended1, attended2, weights1, weights2


class AdaptiveAttention(nn.Module):
    """Adaptive attention with learnable temperature"""

    def __init__(self, feature_dim):
        super(AdaptiveAttention, self).__init__()
        self.feature_dim = feature_dim

        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, query, key, value):
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.temperature + 1e-8)
        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, V)

        return output, attention_weights
