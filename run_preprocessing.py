# ============================================================
# GraphECGNet - Preprocessing Pipeline
# ============================================================
# Run this ONCE to process the CSV signals into Graph Datasets.
# After this finishes, save the /kaggle/working/GraphData folder
# as a new Kaggle Dataset so you don't have to run this again!
# ============================================================

import os
import sys
import argparse

parser = argparse.ArgumentParser(description='GraphECGNet Preprocessing')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to the MIT-BIH dataset on Kaggle')
parser.add_argument('--code_dir', type=str, default='/kaggle/working/GraphECGNet_Kaggle',
                    help='Path to the code directory')
parser.add_argument('--work_dir', type=str, default='/kaggle/working',
                    help='Writable working directory')

args = parser.parse_args()

CODE_DIR = args.code_dir
WORK_DIR = args.work_dir
DATA_PATH = args.data_path

def step0_setup():
    print("=" * 60)
    print("STEP 0: Installing dependencies")
    print("=" * 60)
    os.system('pip install torch_geometric wfdb pandas')
    sys.path.insert(0, CODE_DIR)
    print(f"Data path: {DATA_PATH}")

def step1_signal_to_image():
    print("\n" + "=" * 60)
    print("STEP 1: Converting ECG signals to images")
    print("=" * 60)
    cmd = f'python {CODE_DIR}/signal2image.py --data_path "{DATA_PATH}" --output_path {WORK_DIR}/ECG_images/'
    print(f"Running: {cmd}")
    os.system(cmd)

def step2_edge_transformation():
    print("\n" + "=" * 60)
    print("STEP 2: Applying edge detection")
    print("=" * 60)
    # Train
    cmd1 = f'python {CODE_DIR}/edge_transformation.py --source_base {WORK_DIR}/ECG_images/train --dest_base {WORK_DIR}/ECG_edges/train'
    print(f"Running: {cmd1}")
    os.system(cmd1)
    # Validation/Test
    cmd2 = f'python {CODE_DIR}/edge_transformation.py --source_base {WORK_DIR}/ECG_images/validation --dest_base {WORK_DIR}/ECG_edges/validation'
    print(f"Running: {cmd2}")
    os.system(cmd2)

def step3_graph_construction():
    print("\n" + "=" * 60)
    print("STEP 3: Constructing graph datasets")
    print("=" * 60)
    # Build training set
    cmd1 = f'python {CODE_DIR}/Graph_construction.py --edge_base {WORK_DIR}/ECG_edges/train --output_base {WORK_DIR}/GraphData --dataset_name Trainset_Prewitt_v2_64'
    print(f"Running: {cmd1}")
    os.system(cmd1)
    # Build test set
    cmd2 = f'python {CODE_DIR}/Graph_construction.py --edge_base {WORK_DIR}/ECG_edges/validation --output_base {WORK_DIR}/GraphData --dataset_name Testset_Prewitt_v2_64'
    print(f"Running: {cmd2}")
    os.system(cmd2)

if __name__ == '__main__':
    step0_setup()
    step1_signal_to_image()
    step2_edge_transformation()
    step3_graph_construction()
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"The graph datasets are ready in: {WORK_DIR}/GraphData/")
    print("IMPORTANT: Save the Output (or the GraphData folder) as a Kaggle Dataset now!")
    print("Next time, you can just load that dataset and run the model directly.")
