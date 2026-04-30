import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Kaggle
import matplotlib.pyplot as plt
import random
import cv2
from os.path import isfile, join
from os import listdir
import os
import numpy as np
import time
import glob
import argparse
import pandas as pd
from tqdm import tqdm

# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser(description='Convert ECG signals to images')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to the MIT-BIH dataset (CSV format)')
parser.add_argument('--output_path', type=str, default='/kaggle/working/ECG_images/',
                    help='Path to save generated images')
# ============================================================

_range_to_ignore = 20
_split_percentage = .70
size = 64

# Annotation symbol to class label mapping
labels_json = '{ ".": "NOR", "N": "NOR", "V": "PVC", "/": "PAB", "L": "LBB", "R": "RBB", "A": "APC", "!": "VFW", "E": "VEB" }'
labels_to_float = '{ "NOR": "0", "PVC" : "1", "PAB": "2", "LBB": "3", "RBB": "4", "APC": "5", "VFW": "6", "VEB": "7" }'
original_labels = json.loads(labels_json)
labels = json.loads(labels_to_float)


def parse_annotations(ann_file):
    """
    Parse annotation file to extract sample numbers and beat types.
    Handles multiple annotation file formats.
    """
    samples = []
    symbols = []
    
    with open(ann_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            
            # Try to find sample number and symbol
            # Common format: Time  Sample#  Type  Sub  Chan  Num
            # Example: 0:00.069    25   N    0    0    0
            try:
                # Try second column as sample number
                sample_num = int(parts[1])
                symbol = parts[2]
                samples.append(sample_num)
                symbols.append(symbol)
            except (ValueError, IndexError):
                # Try first column as sample number
                try:
                    sample_num = int(parts[0])
                    symbol = parts[1]
                    samples.append(sample_num)
                    symbols.append(symbol)
                except (ValueError, IndexError):
                    continue
    
    return np.array(samples), symbols


def read_csv_signal(csv_file):
    """
    Read ECG signal from CSV file.
    Format: number_of_sample, raw_value_signal_1, raw_value_signal_2
    """
    try:
        # Try reading with header
        df = pd.read_csv(csv_file)
        if df.shape[1] >= 2:
            # Use first signal column (index 1)
            signal = df.iloc[:, 1].values.astype(float)
            return signal
    except Exception:
        pass
    
    try:
        # Try reading without header
        df = pd.read_csv(csv_file, header=None)
        if df.shape[1] >= 2:
            signal = df.iloc[:, 1].values.astype(float)
            return signal
    except Exception:
        pass
    
    return None


def create_images_from_csv(data_dir, output_dir, img_size=(64, 64)):
    """
    Convert ECG signals from CSV files to grayscale images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files (these are the signal files)
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    
    if len(csv_files) == 0:
        print("ERROR: No .csv files found in " + data_dir)
        print("Files found:", os.listdir(data_dir)[:20])
        return
    
    print(f"Found {len(csv_files)} CSV signal files")
    
    # Split into train/validation
    random.shuffle(csv_files)
    split_point = int(len(csv_files) * _split_percentage)
    train_files = set(csv_files[:split_point])
    val_files = set(csv_files[split_point:])
    
    print(f"Train files: {len(train_files)}, Validation files: {len(val_files)}")
    
    total_images = 0
    
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        # Get record number (e.g., "108" from "108.csv")
        record_num = csv_file.replace('.csv', '')
        
        # Find matching annotation file
        ann_file = None
        for ann_name in [f'{record_num}annotations.txt', f'{record_num}_annotations.txt',
                         f'{record_num}.txt', f'{record_num}ann.txt']:
            candidate = os.path.join(data_dir, ann_name)
            if os.path.exists(candidate):
                ann_file = candidate
                break
        
        if ann_file is None:
            print(f"WARNING: No annotation file found for {csv_file}, skipping.")
            continue
        
        # Read signal
        signal = read_csv_signal(os.path.join(data_dir, csv_file))
        if signal is None:
            print(f"WARNING: Could not read signal from {csv_file}, skipping.")
            continue
        
        # Read annotations
        ann_samples, ann_symbols = parse_annotations(ann_file)
        if len(ann_samples) == 0:
            print(f"WARNING: No annotations found in {ann_file}, skipping.")
            continue
        
        # Determine train or validation
        split = 'train' if csv_file in train_files else 'validation'
        
        # Generate images for each beat
        for i in range(1, len(ann_samples) - 1):
            symbol = ann_symbols[i]
            
            if symbol not in original_labels:
                continue
            
            label = original_labels[symbol]
            
            # Get beat boundaries
            start = ann_samples[i - 1] + _range_to_ignore
            end = ann_samples[i + 1] - _range_to_ignore
            
            if start >= end or start < 0 or end > len(signal):
                continue
            
            # Extract the beat segment
            beat_signal = signal[start:end]
            
            if len(beat_signal) < 10:
                continue
            
            # Create output directory for this class
            class_dir = os.path.join(output_dir, split, label)
            os.makedirs(class_dir, exist_ok=True)
            
            # Plot and save
            fig = plt.figure(frameon=False)
            plt.plot(range(len(beat_signal)), beat_signal)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            
            filename = os.path.join(class_dir, f'{label}_{record_num}_{start}_{end}.png')
            fig.savefig(filename, bbox_inches='tight')
            
            # Convert to grayscale and resize
            im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if im_gray is not None:
                im_gray = cv2.resize(im_gray, img_size, interpolation=cv2.INTER_LANCZOS4)
                im_gray = np.invert(im_gray)
                cv2.imwrite(filename, im_gray)
                total_images += 1
            
            plt.cla()
            plt.clf()
            plt.close('all')
    
    print(f"\nTotal images generated: {total_images}")
    
    # Print summary per class
    for split in ['train', 'validation']:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            print(f"\n{split.upper()} set:")
            for cls in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, cls)
                if os.path.isdir(cls_dir):
                    count = len(os.listdir(cls_dir))
                    print(f"  {cls}: {count} images")


if __name__ == '__main__':
    args = parser.parse_args()
    
    data_dir = args.data_path if args.data_path.endswith('/') else args.data_path + '/'
    output_dir = args.output_path if args.output_path.endswith('/') else args.output_path + '/'
    
    print("Starting signal to image conversion...")
    print(f"Reading from: {data_dir}")
    print(f"Saving to: {output_dir}")
    
    start = time.time()
    create_images_from_csv(data_dir, output_dir)
    elapsed = (time.time() - start) / 60
    print(f"\nSignal to image conversion complete! Time: {elapsed:.2f} min")
