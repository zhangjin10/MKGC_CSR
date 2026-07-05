# MKGC-CSR Implementation

A complete PyTorch implementation of the MKGC-CSR framework from the Scientific Reports paper.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run simple example
python example.py

# Train on your dataset
python main.py --data_path ./data/FB15k-237-IMG --dataset_name FB15k-237-IMG
```

## Project Structure

- `model.py` - Core MKGC-CSR model implementation
- `data_loader.py` - Data loading and preprocessing
- `trainer.py` - Training loop and optimization
- `evaluation.py` - Evaluation metrics and robustness testing
- `feature_extraction.py` - BERT and ViT feature extraction
- `visualization.py` - Visualization utilities
- `ablation_study.py` - Ablation experiments
- `config.py` - Configuration settings
- `main.py` - Main training script
- `example.py` - Simple usage example

## Key Features

✅ Full implementation of causal deconfounding framework
✅ Visual concept dictionary with K-Means clustering
✅ Stratified causal attention mechanism
✅ Robustness evaluation under noise
✅ Ablation study tools
✅ Comprehensive visualization utilities

## Citation

```bibtex
@article{li2026mkgc,
  title={Causal Deconfounding for Robust Link Prediction in Multimodal Knowledge Networks},
  author={Li, Binghong and Xiao, Chuxin},
  journal={Scientific Reports},
  year={2026}
}
```

For more details, see README.md
