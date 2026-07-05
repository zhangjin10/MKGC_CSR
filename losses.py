"""
Loss functions for knowledge graph completion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankingLoss(nn.Module):
    """Margin-based ranking loss"""

    def __init__(self, margin=1.0):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, positive_scores, negative_scores):
        return F.relu(self.margin - positive_scores + negative_scores).mean()


class SelfAdversarialLoss(nn.Module):
    """Self-adversarial negative sampling loss"""

    def __init__(self, margin=1.0, temperature=1.0):
        super(SelfAdversarialLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, positive_scores, negative_scores):
        weights = F.softmax(negative_scores * self.temperature, dim=-1).detach()
        weighted_negative = (negative_scores * weights).sum(dim=-1)

        loss = F.relu(self.margin - positive_scores + weighted_negative).mean()
        return loss


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss for link prediction"""

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, scores, labels):
        return self.criterion(scores, labels)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for multimodal alignment"""

    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        batch_size = features1.size(0)

        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)

        similarity = torch.matmul(features1, features2.T) / self.temperature

        labels = torch.arange(batch_size).to(features1.device)
        loss = F.cross_entropy(similarity, labels)

        return loss


class CausalRegularizationLoss(nn.Module):
    """Causal regularization loss for visual deconfounding"""

    def __init__(self, lambda_reg=0.01):
        super(CausalRegularizationLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, deconfounded_features, raw_features):
        # Encourage deconfounded features to be close but not identical to raw
        reconstruction_loss = F.mse_loss(deconfounded_features, raw_features)

        # Diversity loss to prevent collapse
        batch_size = deconfounded_features.size(0)
        gram_matrix = torch.matmul(deconfounded_features, deconfounded_features.T)
        diversity_loss = -torch.mean(torch.abs(gram_matrix - torch.eye(batch_size).to(gram_matrix.device)))

        total_loss = reconstruction_loss + self.lambda_reg * diversity_loss
        return total_loss
