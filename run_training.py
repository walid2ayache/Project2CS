# ============================================================
# GraphECGNet - Model Training Pipeline
# ============================================================
# Run this for fast iteration on your model design.
# It skips preprocessing and loads the GraphData directly.
# ============================================================

import os
import sys
import argparse
import shutil

parser = argparse.ArgumentParser(description='GraphECGNet Model Training')

# ── Paths ──────────────────────────────────────────────────
parser.add_argument('--graph_data', type=str, required=True,
                    help='Path to the PREPROCESSED GraphData dataset on Kaggle')
parser.add_argument('--code_dir', type=str, default='/kaggle/working/GraphECGNet_Kaggle',
                    help='Path to the code directory')
parser.add_argument('--work_dir', type=str, default='/kaggle/working',
                    help='Writable working directory')

# ── Training schedule ──────────────────────────────────────
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=512,
                    help='Batch size (default: 512)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate (default: 0.001)')
parser.add_argument('--weight_decay', type=float, default=2e-5,
                    help='Weight decay (default: 2e-5)')
parser.add_argument('--step_size', type=int, default=20,
                    help='LR scheduler step size (default: 20)')

# ── Model architecture ─────────────────────────────────────
parser.add_argument('--layer_name', type=str, default='AttentiveSAGE',
                    help='GNN layer type: GCN | GAT | GATv2 | GraphConv | SAGE | AttentiveSAGE')
parser.add_argument('--c_hidden', type=int, default=64,
                    help='Hidden channel dimension (default: 64)')
parser.add_argument('--num_layers', type=int, default=3,
                    help='Number of GNN layers (default: 3)')
parser.add_argument('--dp_rate', type=float, default=0.3,
                    help='Dropout inside GNN backbone (default: 0.3)')
parser.add_argument('--dp_rate_linear', type=float, default=0.5,
                    help='Dropout in the classifier head (default: 0.5)')

args = parser.parse_args()

CODE_DIR  = args.code_dir
WORK_DIR  = args.work_dir
GRAPH_DATA = args.graph_data


def step0_setup():
    print("=" * 60)
    print("STEP 0: Installing dependencies")
    print("=" * 60)
    os.system('pip install torch_geometric')
    sys.path.insert(0, CODE_DIR)
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


def step4_train_model():
    print("\n" + "=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)

    # PyTorch Geometric requires a writable directory to create its 'processed' folder.
    # Since Kaggle input is read-only, we must copy the dataset to /kaggle/working/ first.
    working_graph_data = os.path.join(WORK_DIR, "GraphData")
    print(f"Copying graph dataset from {GRAPH_DATA} to {working_graph_data}...")
    shutil.copytree(GRAPH_DATA, working_graph_data, dirs_exist_ok=True)

    print("\n" + "=" * 60)
    print("TRAINING GNN MODEL")
    print("=" * 60)

    cmd = (
        f'python {CODE_DIR}/main.py'
        f' --root "{working_graph_data}/"'
        f' --output_dir "{WORK_DIR}/"'
        f' --epochs {args.epochs}'
        f' --batch_size {args.batch_size}'
        f' --lr {args.lr}'
        f' --weight_decay {args.weight_decay}'
        f' --step_size {args.step_size}'
        f' --layer_name {args.layer_name}'
        f' --c_hidden {args.c_hidden}'
        f' --num_layers {args.num_layers}'
        f' --dp_rate {args.dp_rate}'
        f' --dp_rate_linear {args.dp_rate_linear}'
    )
    print(f"Running: {cmd}")
    os.system(cmd)


if __name__ == '__main__':
    step0_setup()
    step4_train_model()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Results: {WORK_DIR}/results/")
    print(f"Weights: {WORK_DIR}/weights/")
