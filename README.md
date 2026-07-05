# MKGC-CSR: Causal Deconfounding for Robust Link Prediction in Multimodal Knowledge Networks

Official implementation of the paper published in *Scientific Reports*.

## Overview

This repository contains the PyTorch implementation of **MKGC-CSR** (Multimodal Knowledge Graph Completion with Causal Semantic Reasoning), a novel framework that addresses visual bias in multimodal knowledge graph link prediction through causal deconfounding.

### Key Components

- **Visual Concept Dictionary**: Discretizes continuous visual confounder space via K-Means clustering
- **Stratified Causal Attention**: Implements backdoor adjustment for deconfounding
- **Gated Fusion**: Adaptively integrates structural and multimodal features

## Repository Structure

```
MKGC_CSR_Implementation/
├── models/                      # Model architectures
│   ├── model.py                # Main MKGC-CSR model
│   ├── baseline_models.py      # Baseline implementations
│   ├── trainer.py              # Training procedures
│   ├── attention.py            # Attention mechanisms
│   ├── losses.py               # Loss functions
│   ├── causal_inference.py     # Causal modules
│   └── optimizers.py           # Custom optimizers
├── data/                        # Data processing
│   ├── data_loader.py          # Data loading
│   ├── dataset_utils.py        # Dataset utilities
│   └── graph_sampler.py        # Sampling strategies
├── utils/                       # Utilities
│   ├── config.py               # Configurations
│   ├── feature_extraction.py   # Feature extraction
│   ├── visualization.py        # Visualization tools
│   ├── metrics.py              # Evaluation metrics
│   └── logging_utils.py        # Logging utilities
├── experiments/                 # Experiments
│   ├── evaluation.py           # Evaluation procedures
│   ├── ablation_study.py       # Ablation experiments
│   ├── parameter_sensitivity.py # Parameter analysis
│   ├── case_study.py           # Case studies
│   └── compare_baselines.py    # Baseline comparison
├── scripts/                     # Shell scripts
│   ├── run_training.sh         # Training script
│   ├── run_evaluation.sh       # Evaluation script
│   ├── preprocess_data.sh      # Preprocessing
│   └── compare_baselines.sh    # Baseline comparison
├── main.py                      # Main entry point
├── example.py                   # Usage example
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.3 (optional, for GPU acceleration)

## Quick Start

```bash
# Run example
python example.py

# Train model
python main.py --data_path ./data/FB15k-237-IMG --dataset_name FB15k-237-IMG
```

## Datasets

- **FB15k-237-IMG**: 14,541 entities, 237 relations, 272,115 training triplets
- **WN18-IMG**: 40,943 entities, 18 relations, 141,442 training triplets

## Results

| Model | MRR | Hits@1 | Hits@3 | Hits@10 |
|-------|-----|--------|--------|---------|
| TransE | 0.261 | 0.173 | 0.296 | 0.437 |
| RotatE | 0.338 | 0.241 | 0.375 | 0.533 |
| MKGformer | 0.367 | 0.256 | 0.413 | 0.504 |
| LAFA | 0.398 | 0.269 | 0.447 | 0.551 |
| **MKGC-CSR** | **0.405** | **0.278** | **0.456** | **0.562** |

## Citation

```bibtex
@article{li2026mkgc,
  title={Causal Deconfounding for Robust Link Prediction in Multimodal Knowledge Networks},
  author={Li, Binghong and Xiao, Chuxin},
  journal={Scientific Reports},
  year={2026}
}
```

## License

MIT License

## Contact

- Binghong Li: 15384017902@163.com
- Xiangnan University
