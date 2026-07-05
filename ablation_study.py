"""
Ablation study experiments for MKGC-CSR
Test the contribution of each component
"""

import torch
import torch.nn as nn
from model import MKGC_CSR
from data_loader import create_data_loaders
from evaluation import MetricsEvaluator
import argparse
import json


class MKGC_CSR_NoCausal(MKGC_CSR):
    """MKGC-CSR without causal intervention module"""

    def forward(self, head_entities, relations, tail_entities,
                head_text_features, head_visual_features,
                tail_text_features, tail_visual_features):
        # Structural embeddings
        head_structural = self.structural_encoder(head_entities)
        tail_structural = self.structural_encoder(tail_entities)
        relation_embedding = self.structural_encoder.get_relation_embedding(relations)

        # Process head - no causal intervention
        head_text = self.text_projection(head_text_features)
        head_visual = self.visual_projection(head_visual_features)

        # Direct concatenation without causal deconfounding
        head_multimodal = torch.cat([head_text, head_visual], dim=-1)
        head_multimodal = self.multimodal_projection(head_multimodal)
        head_multimodal = self.dropout(head_multimodal)

        head_final = self.gated_fusion(head_structural, head_multimodal)

        # Process tail
        tail_text = self.text_projection(tail_text_features)
        tail_visual = self.visual_projection(tail_visual_features)

        tail_multimodal = torch.cat([tail_text, tail_visual], dim=-1)
        tail_multimodal = self.multimodal_projection(tail_multimodal)
        tail_multimodal = self.dropout(tail_multimodal)

        tail_final = self.gated_fusion(tail_structural, tail_multimodal)

        # RotatE scoring
        score = self.rotate_score(head_final, relation_embedding, tail_final)

        return score


class MKGC_CSR_NoVisual(MKGC_CSR):
    """MKGC-CSR without visual modality"""

    def forward(self, head_entities, relations, tail_entities,
                head_text_features, head_visual_features,
                tail_text_features, tail_visual_features):
        # Structural embeddings
        head_structural = self.structural_encoder(head_entities)
        tail_structural = self.structural_encoder(tail_entities)
        relation_embedding = self.structural_encoder.get_relation_embedding(relations)

        # Use only text features
        head_text = self.text_projection(head_text_features)
        head_multimodal = torch.cat([head_text, torch.zeros_like(head_text)], dim=-1)
        head_multimodal = self.multimodal_projection(head_multimodal)

        head_final = self.gated_fusion(head_structural, head_multimodal)

        tail_text = self.text_projection(tail_text_features)
        tail_multimodal = torch.cat([tail_text, torch.zeros_like(tail_text)], dim=-1)
        tail_multimodal = self.multimodal_projection(tail_multimodal)

        tail_final = self.gated_fusion(tail_structural, tail_multimodal)

        score = self.rotate_score(head_final, relation_embedding, tail_final)

        return score


class MKGC_CSR_NoText(MKGC_CSR):
    """MKGC-CSR without textual modality"""

    def forward(self, head_entities, relations, tail_entities,
                head_text_features, head_visual_features,
                tail_text_features, tail_visual_features):
        # Structural embeddings
        head_structural = self.structural_encoder(head_entities)
        tail_structural = self.structural_encoder(tail_entities)
        relation_embedding = self.structural_encoder.get_relation_embedding(relations)

        # Use only visual features with causal intervention
        head_visual = self.visual_projection(head_visual_features)
        head_query = head_structural  # Use only structural as query

        head_visual_deconf = self.causal_attention(
            head_query,
            head_visual,
            self.visual_dictionary.get_centroids(),
            self.visual_dictionary.get_priors()
        )

        head_multimodal = torch.cat([torch.zeros_like(head_visual_deconf), head_visual_deconf], dim=-1)
        head_multimodal = self.multimodal_projection(head_multimodal)

        head_final = self.gated_fusion(head_structural, head_multimodal)

        # Similar for tail
        tail_visual = self.visual_projection(tail_visual_features)
        tail_query = tail_structural

        tail_visual_deconf = self.causal_attention(
            tail_query,
            tail_visual,
            self.visual_dictionary.get_centroids(),
            self.visual_dictionary.get_priors()
        )

        tail_multimodal = torch.cat([torch.zeros_like(tail_visual_deconf), tail_visual_deconf], dim=-1)
        tail_multimodal = self.multimodal_projection(tail_multimodal)

        tail_final = self.gated_fusion(tail_structural, tail_multimodal)

        score = self.rotate_score(head_final, relation_embedding, tail_final)

        return score


class MKGC_CSR_UniformPrior(MKGC_CSR):
    """MKGC-CSR with uniform prior instead of empirical prior"""

    def forward(self, head_entities, relations, tail_entities,
                head_text_features, head_visual_features,
                tail_text_features, tail_visual_features):
        # Same as MKGC_CSR but use uniform priors
        head_structural = self.structural_encoder(head_entities)
        tail_structural = self.structural_encoder(tail_entities)
        relation_embedding = self.structural_encoder.get_relation_embedding(relations)

        head_text = self.text_projection(head_text_features)
        head_visual_raw = self.visual_projection(head_visual_features)
        head_query = torch.cat([head_structural, head_text], dim=-1)

        # Use uniform priors
        uniform_priors = torch.ones_like(self.visual_dictionary.get_priors()) / self.visual_dictionary.num_concepts

        head_visual_deconf = self.causal_attention(
            head_query,
            head_visual_raw,
            self.visual_dictionary.get_centroids(),
            uniform_priors
        )

        head_multimodal = torch.cat([head_text, head_visual_deconf], dim=-1)
        head_multimodal = self.multimodal_projection(head_multimodal)
        head_final = self.gated_fusion(head_structural, head_multimodal)

        tail_text = self.text_projection(tail_text_features)
        tail_visual_raw = self.visual_projection(tail_visual_features)
        tail_query = torch.cat([tail_structural, tail_text], dim=-1)

        tail_visual_deconf = self.causal_attention(
            tail_query,
            tail_visual_raw,
            self.visual_dictionary.get_centroids(),
            uniform_priors
        )

        tail_multimodal = torch.cat([tail_text, tail_visual_deconf], dim=-1)
        tail_multimodal = self.multimodal_projection(tail_multimodal)
        tail_final = self.gated_fusion(tail_structural, tail_multimodal)

        score = self.rotate_score(head_final, relation_embedding, tail_final)

        return score


def run_ablation_study(args):
    """Run complete ablation study"""

    print("="*80)
    print("Ablation Study for MKGC-CSR")
    print("="*80)

    # Load data
    train_loader, valid_loader, test_loader, preprocessor = create_data_loaders(
        args.data_path,
        args.dataset_name,
        args.batch_size,
        args.num_workers
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model variants
    model_variants = {
        'Full Model': MKGC_CSR,
        'w/o Causal Intervention': MKGC_CSR_NoCausal,
        'w/o Visual Modality': MKGC_CSR_NoVisual,
        'w/o Textual Modality': MKGC_CSR_NoText,
        'Uniform Prior': MKGC_CSR_UniformPrior
    }

    results = {}

    for variant_name, ModelClass in model_variants.items():
        print(f"\n{'='*80}")
        print(f"Testing: {variant_name}")
        print(f"{'='*80}")

        # Create model
        model = ModelClass(
            num_entities=len(preprocessor.entity2id),
            num_relations=len(preprocessor.relation2id),
            structural_dim=args.structural_dim,
            text_dim=args.text_dim,
            visual_dim=args.visual_dim,
            num_visual_concepts=args.num_visual_concepts
        ).to(device)

        # Load checkpoint (assuming models are pre-trained)
        checkpoint_path = f"{args.checkpoint_dir}/{variant_name.replace(' ', '_').replace('/', '')}_best_model.pt"

        # For demo purposes, evaluate with random initialization
        # In practice, you should train each variant separately

        # Evaluate
        evaluator = MetricsEvaluator(model, preprocessor, device)

        # Convert test data to triplet list
        test_triplets = []
        for batch in test_loader:
            for i in range(len(batch['head'])):
                h = batch['head'][i].item()
                r = batch['relation'][i].item()
                t = batch['tail'][i].item()
                test_triplets.append((h, r, t))

        # Evaluate (use subset for quick demo)
        metrics = evaluator.evaluate_triplets(test_triplets[:100], corrupt_mode='tail')

        results[variant_name] = metrics

        print(f"\nResults for {variant_name}:")
        print(f"  MRR: {metrics['MRR']:.4f}")
        print(f"  Hits@1: {metrics['Hits@1']:.4f}")
        print(f"  Hits@3: {metrics['Hits@3']:.4f}")
        print(f"  Hits@10: {metrics['Hits@10']:.4f}")

    # Print comparison table
    print("\n" + "="*80)
    print("Ablation Study Results Summary")
    print("="*80)
    print(f"{'Variant':<30} {'MRR':<10} {'Hits@1':<10} {'Hits@3':<10} {'Hits@10':<10}")
    print("-"*80)

    for variant_name, metrics in results.items():
        print(f"{variant_name:<30} {metrics['MRR']:<10.4f} {metrics['Hits@1']:<10.4f} "
              f"{metrics['Hits@3']:<10.4f} {metrics['Hits@10']:<10.4f}")

    print("="*80)

    # Save results
    output_path = f"{args.checkpoint_dir}/ablation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nAblation results saved to {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation study for MKGC-CSR')

    parser.add_argument('--data_path', type=str, default='./data/FB15k-237-IMG')
    parser.add_argument('--dataset_name', type=str, default='FB15k-237-IMG')
    parser.add_argument('--structural_dim', type=int, default=200)
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--visual_dim', type=int, default=768)
    parser.add_argument('--num_visual_concepts', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    args = parser.parse_args()

    run_ablation_study(args)
