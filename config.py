"""
Configuration file for MKGC-CSR experiments
"""

# Dataset configurations
DATASET_CONFIG = {
    'FB15k-237-IMG': {
        'num_entities': 14541,
        'num_relations': 237,
        'train_size': 272115,
        'valid_size': 17535,
        'test_size': 20466,
        'images_per_entity': 10
    },
    'WN18-IMG': {
        'num_entities': 40943,
        'num_relations': 18,
        'train_size': 141442,
        'valid_size': 5000,
        'test_size': 5000,
        'images_per_entity': 5
    }
}

# Model configurations
MODEL_CONFIG = {
    'default': {
        'structural_dim': 200,
        'text_dim': 768,
        'visual_dim': 768,
        'num_visual_concepts': 64,
        'dropout': 0.1
    },
    'large': {
        'structural_dim': 400,
        'text_dim': 768,
        'visual_dim': 768,
        'num_visual_concepts': 128,
        'dropout': 0.1
    },
    'small': {
        'structural_dim': 100,
        'text_dim': 768,
        'visual_dim': 768,
        'num_visual_concepts': 32,
        'dropout': 0.1
    }
}

# Training configurations
TRAINING_CONFIG = {
    'default': {
        'batch_size': 512,
        'learning_rate': 1e-4,
        'num_epochs': 500,
        'num_negative': 256,
        'margin': 1.0,
        'patience': 50,
        'warmup_epochs': 10,
        'weight_decay': 0.0,
        'gradient_clip': 1.0
    },
    'fast': {
        'batch_size': 1024,
        'learning_rate': 2e-4,
        'num_epochs': 200,
        'num_negative': 128,
        'margin': 1.0,
        'patience': 30,
        'warmup_epochs': 5,
        'weight_decay': 0.0,
        'gradient_clip': 1.0
    }
}

# Feature extraction configurations
FEATURE_CONFIG = {
    'bert': {
        'model_name': 'bert-base-uncased',
        'max_length': 128,
        'batch_size': 32
    },
    'vit': {
        'model_name': 'google/vit-base-patch16-224',
        'image_size': 224,
        'batch_size': 32
    }
}

# Evaluation configurations
EVAL_CONFIG = {
    'filtered': True,
    'metrics': ['MRR', 'Hits@1', 'Hits@3', 'Hits@10'],
    'batch_size': 64
}

# Robustness test configurations
ROBUSTNESS_CONFIG = {
    'noise_levels': [0.0, 0.1, 0.2, 0.3],
    'noise_type': 'gaussian',
    'test_subset_size': 1000
}

# Visual concept dictionary configurations
VISUAL_DICT_CONFIG = {
    'num_concepts_options': [16, 32, 48, 64, 96, 128],
    'clustering_method': 'kmeans',
    'random_state': 42,
    'n_init': 10
}

# Paths
PATH_CONFIG = {
    'data_root': './data',
    'checkpoint_root': './checkpoints',
    'results_root': './results',
    'visualization_root': './visualizations'
}
