#!/bin/bash

# Set default GPU value
DEFAULT_GPU=0
# Use the first command-line argument as the GPU value, or default to DEFAULT_GPU if not provided
GPU=${1:-$DEFAULT_GPU}

# Set default noise level
DEFAULT_NOISE=0
# Use the second command-line argument as the noise value, or default to DEFAULT_NOISE if not provided
NOISE=${2:-$DEFAULT_NOISE}

# Run the VQSynergy training script with specified arguments
python VQSynergy/train_vqsynergy.py \
    --dataset_name ALMANAC \
    --quantized \
    --noise $NOISE \
    --rd_seed 1 \
    --gpu $GPU \
    --cv_ratio 0.9 \
    --cv_mode_ls 1 2 3 \
    --num_split 5 \
    --max_epoch 2000 \
    --start_update_epoch 599 \
    --print_interval 200 \
    --learning_rate 3e-5 \
    --lr_decay 0.9997 \
    --min_lr 1e-6 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 2e-2 \
    --alpha 1e-2 \
    --initializer_hidden_layers 2 \
    --drug_heads 4 \
    --graph_maxpooling \
    --refiner_in_dim 256 \
    --refiner_out_dim 256 \
    --multiplier 2.0 \
    --refiner_hidden_layers 3 \
    --num_embeddings 1024 \
    --commitment_cost 1e-1 \
    --kmeans \
    --decay 0.99 \
    --lambda_ 1.0 \
    --nu 0.0 \
    --tau 1.0