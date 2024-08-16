import os
import json
import argparse
from datetime import datetime
import torch


def parse_arguments():
    parser = argparse.ArgumentParser("VQSynergy")

    # Dataset and Mode Arguments
    parser.add_argument('--checkpoint', type=str,
                        default=None, help='Checkpoint for the run')
    parser.add_argument('--dataset_name', type=str, default='ALMANAC',
                        choices=['ONEIL', 'ALMANAC'], help='Dataset name')
    parser.add_argument('--quantized', action='store_true',
                        help='Enable vector quantization')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Noise intensity for gene expression data')  # 0.1 or 0.0

    # Environment Arguments
    parser.add_argument('--rd_seed', type=int, default=1, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU number to use if available')
    parser.add_argument('--cv_ratio', type=float, default=0.9,
                        help='Proportion of data to use for cross-validation')
    parser.add_argument('--swap', action='store_true', help='Enable swap mode')

    # Training Arguments
    parser.add_argument('--cv_mode_ls', nargs='+', type=int,
                        default=[], help='Cross-validation modes')
    parser.add_argument('--num_split', type=int,
                        default=5, help='Number of splits')
    parser.add_argument('--max_epoch', type=int, default=2000,
                        help='Maximum number of epochs')
    parser.add_argument('--start_update_epoch', type=int,
                        default=599, help='Epoch to start updating')
    parser.add_argument('--print_interval', type=int,
                        default=200, help='Print interval')

    # Optimization Arguments
    parser.add_argument('--learning_rate', type=float,
                        default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9997,
                        help='Learning rate decay')  # 0.9997 or 1
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 hyperparameter for AdamW optimizer')
    parser.add_argument('--beta2', type=float, default=0.95,
                        help='Beta2 hyperparameter for AdamW optimizer')
    parser.add_argument('--weight_decay', type=float,
                        default=2e-2, help='Weight decay')  # 5e-2
    parser.add_argument('--alpha', type=float, default=1e-2,
                        help='Reconstruction regularization parameter alpha')  # 1e-2 or 0.4

    # Architecture Arguments
    parser.add_argument('--initializer_hidden_layers', type=int,
                        default=2, help='Number of initializer\'s hidden layers')
    parser.add_argument('--drug_heads', type=int, default=4,
                        help='Number of TransformerConv heads in drugs')
    parser.add_argument('--graph_maxpooling', action='store_true',
                        help='Use max pooling in initializer')
    parser.add_argument('--refiner_in_dim', type=int,
                        default=256, help='Input dimension for Refiner')
    parser.add_argument('--refiner_out_dim', type=int,
                        default=256, help='Output dimension for Refiner')
    parser.add_argument('--multiplier', type=float, default=1.0,
                        help='Multiplier to scale the hidden dimension relative to the output dimension in Refiner')
    parser.add_argument('--refiner_hidden_layers', type=int,
                        default=3, help='Number of Refiner\'s hidden layers')

    # Vector Quantizer Arguments
    parser.add_argument('--num_embeddings', type=int, default=1024,
                        help='Number of embeddings in the VQ codebook')
    parser.add_argument('--commitment_cost', type=float, default=0.0,
                        help='Commitment cost for VQ')
    parser.add_argument('--kmeans', action='store_true',
                        help='Use KMeans to initialize the VQ codebook')
    parser.add_argument('--decay', type=float, default=0.99,
                        help='Decay parameter for EMA updates')
    parser.add_argument('--lambda_', type=float, default=0,
                        help='Smoothing parameter for EMA updates')
    parser.add_argument('--nu', type=float, default=0.0,
                        help='Straightening parameter for synchronized VQ')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Temperature parameter for Gumbel Softmax in VQ')

    args = parser.parse_args()

    args.use_checkpoint = args.checkpoint is not None
    if not args.use_checkpoint:
        args.checkpoint = datetime.now().strftime("%m%d_%H%M%S")

    for mode in args.cv_mode_ls:
        if mode not in [1, 2, 3]:
            parser.error(
                f"Invalid value for --cv_mode_ls: {mode}. Allowed values are 1, 2, 3.")

    return args


ARGS = parse_arguments()

if torch.cuda.is_available() and ARGS.gpu >= 0:
    DEVICE = torch.device(f'cuda:{ARGS.gpu}')
    print(f"Using GPU (cuda:{ARGS.gpu})...")
else:
    DEVICE = torch.device('cpu')
    print("Using CPU...")

DATASET_DIR = '../Data'
CHECKPOINT_DIR = '../checkpoints'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, ARGS.checkpoint)
DRUG_DIM = 75
THRESHOLD = 30
CE_CRITERION = torch.nn.BCELoss()

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

with open(os.path.join(CHECKPOINT_PATH, 'args.json'), 'w') as f:
    json.dump(vars(ARGS), f, indent=4)
