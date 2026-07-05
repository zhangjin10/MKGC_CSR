#!/bin/bash

# Compare MKGC-CSR with baseline models

echo "Running baseline comparisons..."

DATA_PATH="./data/FB15k-237-IMG"
CHECKPOINT_DIR="./checkpoints"

# TransE
echo "Evaluating TransE..."
python experiments/compare_baselines.py \
    --model TransE \
    --data_path $DATA_PATH \
    --checkpoint_dir $CHECKPOINT_DIR/transe

# DistMult
echo "Evaluating DistMult..."
python experiments/compare_baselines.py \
    --model DistMult \
    --data_path $DATA_PATH \
    --checkpoint_dir $CHECKPOINT_DIR/distmult

# MKGformer
echo "Evaluating MKGformer..."
python experiments/compare_baselines.py \
    --model MKGformer \
    --data_path $DATA_PATH \
    --checkpoint_dir $CHECKPOINT_DIR/mkgformer

# LAFA
echo "Evaluating LAFA..."
python experiments/compare_baselines.py \
    --model LAFA \
    --data_path $DATA_PATH \
    --checkpoint_dir $CHECKPOINT_DIR/lafa

# MKGC-CSR (ours)
echo "Evaluating MKGC-CSR..."
python experiments/compare_baselines.py \
    --model MKGC_CSR \
    --data_path $DATA_PATH \
    --checkpoint_dir $CHECKPOINT_DIR/mkgc_csr

echo "Comparison completed!"
