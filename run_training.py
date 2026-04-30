# ============================================================
# GraphECGNet - Model Training Pipeline
# ============================================================
# Run this for fast iteration on your model design.
# It skips preprocessing and loads the GraphData directly.
# ============================================================

import os
import sys
import argparse

parser = argparse.ArgumentParser(description='GraphECGNet Model Training')
parser.add_argument('--graph_data', type=str, required=True,
                    help='Path to the PREPROCESSED GraphData dataset on Kaggle')
parser.add_argument('--code_dir', type=str, default='/kaggle/working/GraphECGNet_Kaggle',
                    help='Path to the code directory')
parser.add_argument('--work_dir', type=str, default='/kaggle/working',
                    help='Writable working directory')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')

args = parser.parse_args()

CODE_DIR = args.code_dir
WORK_DIR = args.work_dir
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

import shutil

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
    
    cmd = f'python {CODE_DIR}/main.py --root "{working_graph_data}/" --output_dir "{WORK_DIR}/" --epochs {args.epochs}'
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
