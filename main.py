"""
Main training script for MKGC-CSR
Run this script to train the model on FB15k-237-IMG or WN18-IMG
"""

import torch
import argparse
import json
import os
from model import MKGC_CSR
from data_loader import create_data_loaders
from trainer import MKGC_CSR_Trainer
from evaluation import MetricsEvaluator, RobustnessEvaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train MKGC-CSR model')

    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='./data/FB15k-237-IMG',
                        help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, default='FB15k-237-IMG',
                        choices=['FB15k-237-IMG', 'WN18-IMG'],
                        help='Dataset name')

    # Model parameters
    parser.add_argument('--structural_dim', type=int, default=200,
                        help='Structural embedding dimension')
    parser.add_argument('--text_dim', type=int, default=768,
                        help='Text feature dimension')
    parser.add_argument('--visual_dim', type=int, default=768,
                        help='Visual feature dimension')
    parser.add_argument('--num_visual_concepts', type=int, default=64,
                        help='Number of visual concept prototypes (K)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Maximum number of training epochs')
    parser.add_argument('--num_negative', type=int, default=256,
                        help='Number of negative samples per positive')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for ranking loss')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')

    # Other parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate a trained model')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    parser.add_argument('--run_robustness', action='store_true',
                        help='Run robustness evaluation with noise')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def main():
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Print configuration
    print("="*80)
    print("MKGC-CSR: Multimodal Knowledge Graph Completion with Causal Semantic Reasoning")
    print("="*80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Data path: {args.data_path}")
    print(f"Structural dimension: {args.structural_dim}")
    print(f"Visual concepts: {args.num_visual_concepts}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*80)

    # Create data loaders
    print("\nLoading data...")
    train_loader, valid_loader, test_loader, preprocessor = create_data_loaders(
        args.data_path,
        args.dataset_name,
        args.batch_size,
        args.num_workers
    )

    # Create model
    print("\nInitializing model...")
    model = MKGC_CSR(
        num_entities=len(preprocessor.entity2id),
        num_relations=len(preprocessor.relation2id),
        structural_dim=args.structural_dim,
        text_dim=args.text_dim,
        visual_dim=args.visual_dim,
        num_visual_concepts=args.num_visual_concepts
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training configuration
    config = {
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'num_negative': args.num_negative,
        'margin': args.margin,
        'patience': args.patience,
        'checkpoint_dir': args.checkpoint_dir
    }

    # Create trainer
    trainer = MKGC_CSR_Trainer(
        model, train_loader, valid_loader, test_loader,
        preprocessor, config
    )

    if not args.eval_only:
        # Train model
        test_metrics = trainer.train()

        # Save results
        results = {
            'dataset': args.dataset_name,
            'config': config,
            'test_metrics': test_metrics
        }

        results_path = os.path.join(args.checkpoint_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\nResults saved to {results_path}")

    else:
        # Evaluation only
        if args.checkpoint_path is None:
            args.checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')

        print(f"\nLoading checkpoint from {args.checkpoint_path}")
        trainer.load_checkpoint(args.checkpoint_path)

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(test_loader, mode='test')

        print("\n" + "="*80)
        print("Test Results")
        print("="*80)
        print(f"MRR: {test_metrics['MRR']:.4f}")
        print(f"Hits@1: {test_metrics['Hits@1']:.4f}")
        print(f"Hits@3: {test_metrics['Hits@3']:.4f}")
        print(f"Hits@10: {test_metrics['Hits@10']:.4f}")
        print("="*80)

    # Run robustness analysis if requested
    if args.run_robustness:
        print("\n" + "="*80)
        print("Running Robustness Analysis")
        print("="*80)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        robustness_eval = RobustnessEvaluator(model, preprocessor, device)

        # Convert test data to triplet list
        test_triplets = []
        for batch in test_loader:
            for i in range(len(batch['head'])):
                h = batch['head'][i].item()
                r = batch['relation'][i].item()
                t = batch['tail'][i].item()
                test_triplets.append((h, r, t))

        # Run robustness analysis
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        robustness_results = robustness_eval.robustness_analysis(
            test_triplets[:1000],  # Use subset for faster evaluation
            noise_levels
        )

        # Save robustness results
        robustness_path = os.path.join(args.checkpoint_dir, 'robustness_results.json')
        with open(robustness_path, 'w') as f:
            json.dump({k: v for k, v in robustness_results.items()}, f, indent=4)

        print(f"\nRobustness results saved to {robustness_path}")


if __name__ == '__main__':
    main()
