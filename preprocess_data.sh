#!/bin/bash

# Preprocess data for MKGC-CSR

echo "Preprocessing data for MKGC-CSR..."

DATA_PATH=$1
DATASET_NAME=$2

if [ -z "$DATA_PATH" ]; then
    DATA_PATH="./data/FB15k-237-IMG"
fi

if [ -z "$DATASET_NAME" ]; then
    DATASET_NAME="FB15k-237-IMG"
fi

echo "Data path: $DATA_PATH"
echo "Dataset: $DATASET_NAME"

# Extract features
python utils/feature_extraction.py \
    --data_path $DATA_PATH \
    --dataset_name $DATASET_NAME \
    --device cuda

echo "Preprocessing completed!"
