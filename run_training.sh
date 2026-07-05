#!/bin/bash

# Quick start script for MKGC-CSR training
# This script sets up the environment and runs training on FB15k-237-IMG

echo "=========================================="
echo "MKGC-CSR Quick Start"
echo "=========================================="

# Step 1: Create directories
echo "Creating necessary directories..."
mkdir -p data/FB15k-237-IMG
mkdir -p checkpoints
mkdir -p results
mkdir -p visualizations

# Step 2: Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Step 3: Check for dataset
if [ ! -f "data/FB15k-237-IMG/train.txt" ]; then
    echo "WARNING: Dataset not found in data/FB15k-237-IMG/"
    echo "Please download FB15k-237-IMG dataset and place it in the data directory"
    echo "Expected files:"
    echo "  - train.txt"
    echo "  - valid.txt"
    echo "  - test.txt"
    echo "  - entity_descriptions.json"
    echo "  - entity_images.json"
    exit 1
fi

# Step 4: Extract features (if not already done)
if [ ! -f "data/FB15k-237-IMG/text_features.pt" ] || [ ! -f "data/FB15k-237-IMG/visual_features.pt" ]; then
    echo "Extracting multimodal features..."
    python feature_extraction.py \
        --data_path ./data/FB15k-237-IMG \
        --dataset_name FB15k-237-IMG \
        --device cuda
else
    echo "Features already extracted, skipping..."
fi

# Step 5: Train model
echo "Starting training..."
python main.py \
    --data_path ./data/FB15k-237-IMG \
    --dataset_name FB15k-237-IMG \
    --structural_dim 200 \
    --num_visual_concepts 64 \
    --batch_size 512 \
    --learning_rate 1e-4 \
    --num_epochs 500 \
    --patience 50 \
    --checkpoint_dir ./checkpoints

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to ./checkpoints/"
echo "=========================================="
