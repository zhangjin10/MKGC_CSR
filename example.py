"""
Simple example demonstrating MKGC-CSR usage
"""

import torch
from model import MKGC_CSR
import numpy as np


def simple_example():
    """Simple example of using MKGC-CSR for link prediction"""

    print("="*80)
    print("MKGC-CSR Simple Example")
    print("="*80)

    # Model parameters
    num_entities = 100
    num_relations = 10
    structural_dim = 200
    text_dim = 768
    visual_dim = 768
    num_visual_concepts = 64

    # Create model
    print("\nCreating MKGC-CSR model...")
    model = MKGC_CSR(
        num_entities=num_entities,
        num_relations=num_relations,
        structural_dim=structural_dim,
        text_dim=text_dim,
        visual_dim=visual_dim,
        num_visual_concepts=num_visual_concepts
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create sample data
    batch_size = 4

    # Sample triplets
    head_entities = torch.randint(0, num_entities, (batch_size,))
    relations = torch.randint(0, num_relations, (batch_size,))
    tail_entities = torch.randint(0, num_entities, (batch_size,))

    # Sample features
    head_text_features = torch.randn(batch_size, text_dim)
    head_visual_features = torch.randn(batch_size, visual_dim)
    tail_text_features = torch.randn(batch_size, text_dim)
    tail_visual_features = torch.randn(batch_size, visual_dim)

    print("\nSample input:")
    print(f"  Head entities: {head_entities.tolist()}")
    print(f"  Relations: {relations.tolist()}")
    print(f"  Tail entities: {tail_entities.tolist()}")

    # Forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        scores = model(
            head_entities, relations, tail_entities,
            head_text_features, head_visual_features,
            tail_text_features, tail_visual_features
        )

    print(f"\nPredicted scores: {scores.numpy()}")
    print(f"Score shape: {scores.shape}")

    # Demonstrate visual concept dictionary
    print("\n" + "="*80)
    print("Visual Concept Dictionary")
    print("="*80)

    visual_dict = model.visual_dictionary
    print(f"Number of concepts: {visual_dict.num_concepts}")
    print(f"Centroids shape: {visual_dict.centroids.shape}")
    print(f"Priors: {visual_dict.priors[:10].numpy()}")  # Show first 10

    # Demonstrate causal attention
    print("\n" + "="*80)
    print("Causal Attention Mechanism")
    print("="*80)

    # Create sample query and visual features
    query = torch.randn(batch_size, structural_dim * 2 + text_dim)
    visual_raw = torch.randn(batch_size, visual_dim)

    with torch.no_grad():
        deconfounded_visual = model.causal_attention(
            query,
            visual_raw,
            visual_dict.get_centroids(),
            visual_dict.get_priors()
        )

    print(f"Input visual features shape: {visual_raw.shape}")
    print(f"Deconfounded visual features shape: {deconfounded_visual.shape}")
    print(f"\nL2 norm of input: {visual_raw.norm(dim=1).numpy()}")
    print(f"L2 norm of output: {deconfounded_visual.norm(dim=1).numpy()}")

    # Demonstrate link prediction
    print("\n" + "="*80)
    print("Link Prediction Example")
    print("="*80)

    # Predict tail given head and relation
    head = torch.tensor([5])
    relation = torch.tensor([2])
    head_text = torch.randn(1, text_dim)
    head_visual = torch.randn(1, visual_dim)

    print(f"\nQuery: (head={head.item()}, relation={relation.item()}, tail=?)")

    # Score all possible tails
    tail_scores = []
    with torch.no_grad():
        for tail_candidate in range(min(20, num_entities)):  # Score first 20 entities
            tail = torch.tensor([tail_candidate])
            tail_text = torch.randn(1, text_dim)
            tail_visual = torch.randn(1, visual_dim)

            score = model(
                head, relation, tail,
                head_text, head_visual,
                tail_text, tail_visual
            )
            tail_scores.append((tail_candidate, score.item()))

    # Sort by score
    tail_scores.sort(key=lambda x: x[1], reverse=True)

    print("\nTop-5 predicted tails:")
    for rank, (tail_id, score) in enumerate(tail_scores[:5], 1):
        print(f"  {rank}. Tail {tail_id}: score = {score:.4f}")

    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == '__main__':
    simple_example()
