import cv2
import os
import glob
import time
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Build graph datasets from edge images')
parser.add_argument('--edge_base', type=str, required=True,
                    help='Base directory of edge-detected images (with class subfolders)')
parser.add_argument('--output_base', type=str, default='/kaggle/working/GraphData',
                    help='Base directory to save graph .txt files')
parser.add_argument('--dataset_name', type=str, default='Trainset_Prewitt_v2_224',
                    help='Name for the dataset')

# Global variables for graph construction
edges           = []
attrs           = []
graph_id        = 1
node_id         = 1
graph_indicator = []
node_labels     = []
graph_labels    = []

# ── CHANGE (Option C) ────────────────────────────────────────────────────────
# Reduced from 8 classes to 7. VFW (6) and VEB (7) are replaced by OTH (6).
# The integer label 6 now means "Other/Unknown beat" — a merged class that
# pools the former VFW and VEB samples. The old integer labels 6 and 7 no
# longer exist; only 0–6 are valid.
# ─────────────────────────────────────────────────────────────────────────────
activity_map   = {0: 'NOR', 1: 'PVC', 2: 'PAB', 3: 'LBB', 4: 'RBB', 5: 'APC', 6: 'OTH'}
class_to_label = {v: k for k, v in activity_map.items()}

# ── CHANGE (Option C) ────────────────────────────────────────────────────────
# ECG_CLASSES now lists 7 folder names. The edge_transformation step will have
# produced an "OTH" subfolder (containing the former VFW + VEB images).
# ─────────────────────────────────────────────────────────────────────────────
ECG_CLASSES_DEFAULT = ['NOR', 'PVC', 'PAB', 'LBB', 'RBB', 'APC', 'OTH']


def normalize(arr):
    arr = np.array(arr)
    m, s = np.mean(arr), np.std(arr)
    return (arr - m) / s if s != 0 else arr - m


def generate_graphs(filename, node_label):
    global node_id, edges, attrs, graph_id, node_labels, graph_indicator
    print(" ... Reading image: " + filename + " ...")
    cnt = 0
    img = cv2.imread(filename)
    if img is None:
        print(f"WARNING: Could not read {filename}, skipping.")
        return
    dim1, dim2, _ = img.shape
    attrs1 = []
    nodes  = np.full((dim1, dim2), -1)
    edge   = 0

    for i in range(dim1):
        for j in range(dim2):
            b, _, _ = img[i][j]
            if b >= 128:
                nodes[i][j] = node_id
                attrs1.append(b)
                graph_indicator.append(graph_id)
                node_labels.append([node_label, activity_map[node_label]])
                node_id += 1
                cnt += 1

    for i in range(dim1):
        for j in range(dim2):
            if nodes[i][j] != -1:
                for i1 in range(max(0, i-1), min(i+2, dim1)):
                    for j1 in range(max(0, j-1), min(j+2, dim2)):
                        if (i1 != i or j1 != j) and nodes[i1][j1] != -1:
                            edges.append([nodes[i][j], nodes[i1][j1]])
                            edge += 1

    attrs1 = normalize(attrs1)
    attrs.extend(attrs1)
    print(f"Nodes: {cnt}, Edges: {edge}")
    if cnt != 0:
        graph_id += 1


def generate_graph_with_labels(dirname, label):
    global graph_labels
    print(f"\n... Reading Directory: {dirname} ...\n")
    for filename in glob.glob(dirname + '/*.png'):
        generate_graphs(filename, label)
        graph_labels.append([label, activity_map[label]])


def save_dataframe_to_txt(df, filepath):
    df.to_csv(filepath, header=None, index=None, sep=',', mode='w')


def build_graph_dataset(edge_base, output_base, dataset_name, classes):
    global edges, attrs, graph_id, node_id, graph_indicator, node_labels, graph_labels
    edges, attrs, graph_indicator, node_labels, graph_labels = [], [], [], [], []
    graph_id, node_id = 1, 1

    for cls_name in classes:
        cls_dir = os.path.join(edge_base, cls_name)
        if not os.path.exists(cls_dir):
            print(f"WARNING: {cls_dir} not found, skipping.")
            continue
        generate_graph_with_labels(cls_dir, class_to_label[cls_name])

    if len(edges) == 0:
        print("ERROR: No graphs generated!")
        return

    df_A  = pd.DataFrame(data=np.array(edges),       columns=["node-1", "node-2"])
    df_nl = pd.DataFrame(data=np.array(node_labels),  columns=["label", "name"]).drop("name", axis=1)
    df_gl = pd.DataFrame(data=np.array(graph_labels), columns=["label", "name"]).drop("name", axis=1)
    df_na = pd.DataFrame(data=np.array(attrs),        columns=["gray-val"])
    df_gi = pd.DataFrame(data=np.array(graph_indicator), columns=["graph-id"])

    sourcepath = os.path.join(output_base, dataset_name, 'raw')
    os.makedirs(sourcepath, exist_ok=True)

    save_dataframe_to_txt(df_A,  os.path.join(sourcepath, f'{dataset_name}_A.txt'))
    save_dataframe_to_txt(df_gi, os.path.join(sourcepath, f'{dataset_name}_graph_indicator.txt'))
    save_dataframe_to_txt(df_gl, os.path.join(sourcepath, f'{dataset_name}_graph_labels.txt'))
    save_dataframe_to_txt(df_na, os.path.join(sourcepath, f'{dataset_name}_node_attributes.txt'))
    save_dataframe_to_txt(df_nl, os.path.join(sourcepath, f'{dataset_name}_node_labels.txt'))
    print(f"Dataset '{dataset_name}' saved to {sourcepath}")


if __name__ == '__main__':
    args  = parser.parse_args()
    start = time.time()
    build_graph_dataset(args.edge_base, args.output_base, args.dataset_name, ECG_CLASSES_DEFAULT)
    end   = time.time()
    print(f"Total time (min): {(end-start)/60:.2f}")
