"""
Baseline comparison experiments
"""

import torch
import argparse
from models import TransE, DistMult, MKGformer, LAFA, MKGC_CSR
from data import create_data_loaders
from experiments import MetricsEvaluator


def get_model(model_name, num_entities, num_relations, config):
    """Get model by name"""

    if model_name == 'TransE':
        return TransE(num_entities, num_relations, config['embedding_dim'])

    elif model_name == 'DistMult':
        return DistMult(num_entities, num_relations, config['embedding_dim'])

    elif model_name == 'MKGformer':
        return MKGformer(num_entities, num_relations,
                        config['embedding_dim'],
                        config['text_dim'],
                        config['visual_dim'])

    elif model_name == 'LAFA':
        return LAFA(num_entities, num_relations,
                   config['embedding_dim'],
                   config['text_dim'],
                   config['visual_dim'])

    elif model_name == 'MKGC_CSR':
        return MKGC_CSR(num_entities, num_relations,
                       config['structural_dim'],
                       config['text_dim'],
                       config['visual_dim'],
                       config['num_visual_concepts'])

    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=512)

    args = parser.parse_args()

    print(f"Evaluating {args.model}...")

    # Load data
    _, _, test_loader, preprocessor = create_data_loaders(
        args.data_path, batch_size=args.batch_size
    )

    # Model config
    config = {
        'embedding_dim': 200,
        'structural_dim': 200,
        'text_dim': 768,
        'visual_dim': 768,
        'num_visual_concepts': 64
    }

    # Create model
    model = get_model(
        args.model,
        len(preprocessor.entity2id),
        len(preprocessor.relation2id),
        config
    )

    # Load checkpoint if exists
    checkpoint_path = f"{args.checkpoint_dir}/best_model.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Evaluate
    evaluator = MetricsEvaluator(model, preprocessor, device)

    test_triplets = []
    for batch in test_loader:
        for i in range(len(batch['head'])):
            h = batch['head'][i].item()
            r = batch['relation'][i].item()
            t = batch['tail'][i].item()
            test_triplets.append((h, r, t))

    metrics = evaluator.evaluate_triplets(test_triplets[:1000])

    print(f"\n{args.model} Results:")
    print(f"  MRR: {metrics['MRR']:.4f}")
    print(f"  Hits@1: {metrics['Hits@1']:.4f}")
    print(f"  Hits@3: {metrics['Hits@3']:.4f}")
    print(f"  Hits@10: {metrics['Hits@10']:.4f}")


if __name__ == '__main__':
    main()
