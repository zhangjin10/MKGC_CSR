#!/bin/bash

# Evaluation script for MKGC-CSR
# Runs comprehensive evaluation including robustness tests

echo "=========================================="
echo "MKGC-CSR Evaluation"
echo "=========================================="

# Check for checkpoint
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo "ERROR: No trained model found at checkpoints/best_model.pt"
    echo "Please train the model first using run_training.sh"
    exit 1
fi

# Standard evaluation
echo "Running standard evaluation..."
python main.py \
    --data_path ./data/FB15k-237-IMG \
    --dataset_name FB15k-237-IMG \
    --eval_only \
    --checkpoint_path ./checkpoints/best_model.pt

# Robustness evaluation
echo "Running robustness evaluation..."
python main.py \
    --data_path ./data/FB15k-237-IMG \
    --dataset_name FB15k-237-IMG \
    --eval_only \
    --checkpoint_path ./checkpoints/best_model.pt \
    --run_robustness

# Ablation study
echo "Running ablation study..."
python ablation_study.py \
    --data_path ./data/FB15k-237-IMG \
    --dataset_name FB15k-237-IMG \
    --checkpoint_dir ./checkpoints

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to ./checkpoints/"
echo "=========================================="
