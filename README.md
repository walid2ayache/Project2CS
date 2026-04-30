# GraphECGNet - Kaggle Ready Version

This project has been split into **two separate stages** so you only have to run the heavy data preprocessing once!

## 🚀 The 2-Stage Kaggle Workflow

### Stage 1: Data Preprocessing (Run Once)
This stage converts the raw CSV signals into images, applies edge detection, and builds the PyTorch Geometric graph datasets.

1. **Upload this code:** Upload the `GraphECGNet_Kaggle` folder as a Kaggle Dataset (e.g., `graphecgnet-code`).
2. **Create Notebook 1:** Create a new Kaggle Notebook.
3. **Attach Datasets:** Attach your code dataset and the raw CSV dataset (`mondejar/mitbih-database`).
4. **Run Preprocessing:**
```python
# Cell 1: Copy code to writable directory
import shutil, sys
import os

# If Kaggle nested your folder, copy from the inner folder
source_dir = '/kaggle/input/graphecgnet-code/GraphECGNet_Kaggle'
# If it's not nested, uncomment the line below:
# source_dir = '/kaggle/input/graphecgnet-code'

shutil.copytree(source_dir, '/kaggle/working/GraphECGNet_Kaggle', dirs_exist_ok=True)
sys.path.insert(0, '/kaggle/working/GraphECGNet_Kaggle')

# Cell 2: Run the Preprocessing Pipeline
!python /kaggle/working/GraphECGNet_Kaggle/run_preprocessing.py \
    --data_path /kaggle/input/mondejar-mitbih-database  # Adjust to actual dataset path
```
5. **Save Output:** When it finishes, you will see a `/kaggle/working/GraphData` folder. Click **"Save Version"** -> **"Save & Run All (Commit)"**. Then, go to the notebook output and create a **New Dataset** from the `GraphData` output (e.g., name it `graphecgnet-preprocessed-data`).

---

### Stage 2: Model Training (Run Many Times)
This stage skips all preprocessing and directly trains the GNN on your saved graphs. You can restart this notebook as many times as you want to change model architectures, epochs, etc.

1. **Create Notebook 2:** Create a new notebook for training.
2. **Attach Datasets:** Attach your code dataset (`graphecgnet-code`) AND your newly created preprocessed data dataset (`graphecgnet-preprocessed-data`).
3. **Run Training:**
```python
# Cell 1: Copy code to writable directory
import shutil, sys
import os

# If Kaggle nested your folder (e.g. input/dataset/GraphECGNet_Kaggle), copy from the inner folder
source_dir = '/kaggle/input/graphecgnet-code/GraphECGNet_Kaggle'
# If it's not nested, uncomment the line below:
# source_dir = '/kaggle/input/graphecgnet-code'

shutil.copytree(source_dir, '/kaggle/working/GraphECGNet_Kaggle', dirs_exist_ok=True)
sys.path.insert(0, '/kaggle/working/GraphECGNet_Kaggle')

# Cell 2: Run the Training Pipeline
!python /kaggle/working/GraphECGNet_Kaggle/run_training.py \
    --graph_data /kaggle/input/graphecgnet-preprocessed-data/GraphData \
    --epochs 100
```

---

## Modifying the Model
When you want to change the model architecture (like adding layers, changing channels, changing from GCN to GAT), you only need to:
1. Update `models.py` or `main.py` on your computer.
2. Update the `graphecgnet-code` dataset on Kaggle with the new files.
3. Restart **Notebook 2** (Training) and run it! You never have to wait for the images to generate again.

## Available Command-Line Scripts

If you want to run things manually instead of using `run_preprocessing.py` and `run_training.py`:

- **`signal2image.py`**: `--data_path` (Input CSVs), `--output_path` (Output Images)
- **`edge_transformation.py`**: `--source_base` (Input Images), `--dest_base` (Output Edges)
- **`Graph_construction.py`**: `--edge_base` (Input Edges), `--output_base` (Output Graphs), `--dataset_name`
- **`main.py`**: `--root` (Input Graph Data), `--epochs`, `--batch_size`, `--lr`, `--layer_name`
