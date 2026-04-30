#import general packages
import time
import random
import numpy as np
import argparse
import os
import os.path as osp
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Kaggle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

#import torch and PYG
import torch
from torch.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader
from torch_geometric.profile import get_model_size, get_data_size, count_parameters

#import my source code
from models import *
from utils import *
from dataloader import GraphDataset

#Define arguments
parser = argparse.ArgumentParser(description='PYG version of ECG Classification using GNN')

# ============================================================
# KAGGLE PATHS - All defaults set for Kaggle
# ============================================================
parser.add_argument('--root', type=str, default='/kaggle/working/GraphData/', metavar='DIR',
                    help='path to graph dataset root')
parser.add_argument('--training_dataset_name', type=str, default='Trainset_Prewitt_v2_64',
                    help='Choose dataset to train')
parser.add_argument('--testing_dataset_name', type=str, default='Testset_Prewitt_v2_64',
                    help='Choose dataset to test')
parser.add_argument('--output_dir', type=str, default='/kaggle/working/',
                    help='Directory to save weights and results')
# ============================================================

# Hardware and seeds
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA to train a model (default: True)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='choose a random seed (default: 42)')
parser.add_argument('--num_workers', type=int, default=2,
                    help='set number of workers (default: 2)')

# Learning rate schedule
parser.add_argument('-b', '--batch_size', type=int, default=512, metavar='B',
                    help='input batch size for training (default: 512)')
parser.add_argument('--step_size', type=int, default=20, metavar='SS',
                    help='Set step size for scheduler of learning rate (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--weight_decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')

# Model configuration
parser.add_argument('--layer_name', type=str, default='GraphConv',
                    help='choose model type: GAT, GCN, GraphConv, SAGE, AttentiveSAGE')
parser.add_argument('--c_hidden', type=int, default=64,
                    help='Choose numbers of output channels (default: 64)')
parser.add_argument('--num_layers', type=int, default=3,
                    help='Choose numbers of Graph layers for the model (default: 3)')
parser.add_argument('--dp_rate_linear', type=float, default=0.5,
                    help='Set dropout rate at the linear layer (default: 0.5)')
parser.add_argument('--dp_rate', type=float, default=0.5,
                    help='Set dropout rate at every graph layer (default: 0.5)')

# Auto-detect Jupyter vs CLI
try:
    get_ipython()
    args = parser.parse_args(args=[])
    print("Running in Jupyter mode (using default args)")
except NameError:
    args = parser.parse_args()
    print("Running in CLI mode")

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print("***** USE DEVICE *****", device_id, torch.cuda.get_device_name(device_id))
device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
print("==== DEVICE ====", device)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Create output directories
os.makedirs(osp.join(args.output_dir, "weights"), exist_ok=True)
os.makedirs(osp.join(args.output_dir, "results"), exist_ok=True)

#############################
training_dataset = GraphDataset(root=args.root, name=args.training_dataset_name, use_node_attr=True)
testing_dataset  = GraphDataset(root=args.root, name=args.testing_dataset_name,  use_node_attr=True)

print('############## INFORMATION OF TRAINING DATA ##################')
print(f'Training Dataset name: {training_dataset}:')
print(f'Number of graphs:   {len(training_dataset)}')
print(f'Number of features: {training_dataset.num_features}')

# ── CHANGE (Option C) ──────────────────────────────────────────────────────
# After the merge num_classes should report 7 (labels 0-6).
# Print it explicitly so we can verify at runtime.
# ─────────────────────────────────────────────────────────────────────────────
print(f'Number of classes:  {training_dataset.num_classes}  (expected 7 after OTH merge)')

training_data = training_dataset[0]
print(training_data)
print(f'Number of nodes:  {training_data.num_nodes}')
print(f'Number of edges:  {training_data.num_edges}')
print(f'Avg node degree:  {training_data.num_edges / training_data.num_nodes:.2f}')
print(f'Isolated nodes:   {training_data.has_isolated_nodes()}')
print(f'Self-loops:       {training_data.has_self_loops()}')
print(f'Is undirected:    {training_data.is_undirected()}')

print('############## INFORMATION OF TESTING DATA ##################')
print(f'Testing Dataset name: {testing_dataset}:')
print(f'Number of graphs:   {len(testing_dataset)}')
print(f'Number of features: {testing_dataset.num_features}')
print(f'Number of classes:  {testing_dataset.num_classes}')

testing_data = testing_dataset[0]
print(testing_data)
print(f'Number of nodes:  {testing_data.num_nodes}')
print(f'Number of edges:  {testing_data.num_edges}')
print(f'Avg node degree:  {testing_data.num_edges / testing_data.num_nodes:.2f}')

training_dataset = training_dataset.shuffle()

split_idx     = int(len(training_dataset) * 0.1)
train_dataset = training_dataset[split_idx:]
val_dataset   = training_dataset[:split_idx]
test_dataset  = testing_dataset

print(f'Split index:               {split_idx}')
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of val graphs:      {len(val_dataset)}')
print(f'Number of test graphs:     {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}: {data.num_graphs} graphs in batch')
    print(data)
    print()

model = GraphGNNModel(
    c_in=training_dataset.num_node_features,
    c_out=training_dataset.num_classes,
    layer_name=args.layer_name,
    c_hidden=args.c_hidden,
    num_layers=args.num_layers,
    dp_rate_linear=args.dp_rate_linear,
    dp_rate=args.dp_rate
).to(device)

print('Model size:       ', get_model_size(model))
print('Model parameters: ', count_parameters(model))
print(model)
print('Data size:        ', get_data_size(data))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# ── CHANGE (Option C) ──────────────────────────────────────────────────────
# Class weights are still computed dynamically from the training data.
# With 7 classes (0-6) instead of 8, there are no more near-zero counts for
# classes 6 and 7 injecting huge noisy weights.  OTH (class 6) still has fewer
# samples than NOR, so it will get a moderate upward weight — which is correct
# and will no longer destabilise the other class gradients.
# ─────────────────────────────────────────────────────────────────────────────
y_train       = training_dataset.data.y
class_counts  = torch.bincount(y_train)
total_samples = len(y_train)
num_classes   = training_dataset.num_classes

class_weights = torch.sqrt(total_samples / (num_classes * class_counts.float()))
class_weights[class_counts == 0] = 1.0   # safety guard — should not trigger now
class_weights = class_weights.to(device)

print(f"Class counts:  {class_counts.tolist()}")
print(f"Class weights: {class_weights.cpu().numpy()}")
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)


def train():
    model.train()
    for data in tqdm(train_loader, desc=(f'Training epoch: {epoch:04d}')):
        data    = data.to(device)
        out     = model(data.x, data.edge_index, data.batch)
        loss    = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


@torch.no_grad()
def test(loader, desc_name="Validation"):
    model.eval()
    correct     = 0
    y_pred      = []
    y_true      = []
    running_loss = 0
    for data in tqdm(loader, desc=(f'{desc_name} epoch: {epoch:04d}')):
        data    = data.to(device)
        out     = model(data.x, data.edge_index, data.batch)
        loss    = criterion(out, data.y)
        pred    = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(np.squeeze(pred.cpu().numpy().T))
        running_loss += loss.item()
    val_loss = running_loss / len(loader)

    # ── CHANGE (Option C) ──────────────────────────────────────────────────
    # target_names updated to 7 classes; OTH replaces the old VFW / VEB rows.
    # ───────────────────────────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=['NOR', 'PVC', 'PAB', 'LBB', 'RBB', 'APC', 'OTH'],
        digits=4, zero_division=0
    )
    print(report)
    return correct / len(loader.dataset), val_loss


start = time.time()

# ── CHANGE ───────────────────────────────────────────────────────────────────
# best_val_acc was hardcoded to 0.9, meaning no weights were ever saved unless
# the model hit 90% accuracy immediately — an arbitrary and often unreachable
# bar.  Set to 0.0 so the best checkpoint is always saved and improves over
# training, regardless of the absolute score.
# ─────────────────────────────────────────────────────────────────────────────
best_val_acc = 0.0

train_accs, val_accs, train_losses, val_losses = [], [], [], []

for epoch in range(1, args.epochs):
    train()
    train_acc, train_loss = test(train_loader, desc_name="Training Eval")
    val_acc,   val_loss   = test(val_loader,   desc_name="Validation Eval")
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc    = val_acc
        save_weight_path = osp.join(
            args.output_dir, "weights",
            f"Graph_{args.layer_name}_{args.training_dataset_name}_best.pth"
        )
        print(f'New best model saved (val_acc={best_val_acc:.4f}): {save_weight_path}')
        torch.save(model.state_dict(), save_weight_path)

    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Training curve plot
fig, ax = plt.subplots()
ax.plot(train_accs,   c="steelblue",  label="Train accuracy")
ax.plot(val_accs,     c="orangered",  label="Val accuracy")
ax.plot(train_losses, c="black",      label="Train loss")
ax.plot(val_losses,   c="green",      label="Val loss")
ax.grid()
ax.legend(loc='best')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy / Loss')
ax.set_title("Training evolution")
plt.savefig(osp.join(
    args.output_dir, "results",
    f"Evolution_training_{args.layer_name}_{args.training_dataset_name}.png"
))

end = time.time()
print(f"Total training time (min): {(end - start) / 60:.2f}")
print("****End training process here******")


@torch.no_grad()
def inference(loader):
    model.eval()
    correct = 0
    y_pred  = []
    y_true  = []
    for data in loader:
        data    = data.to(device)
        out     = model(data.x, data.edge_index, data.batch)
        pred    = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(np.squeeze(pred.cpu().numpy().T))

    # ── CHANGE (Option C) ──────────────────────────────────────────────────
    # 7 labels: NOR, PVC, PAB, LBB, RBB, APC, OTH  (integer labels 0-6).
    # Confusion matrix now has 7 rows/columns instead of 8.
    # ───────────────────────────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=['NOR', 'PVC', 'PAB', 'LBB', 'RBB', 'APC', 'OTH'],
        digits=4, zero_division=0
    )
    print(report)

    display_labels = ['NOR', 'PVC', 'PAB', 'LBB', 'RBB', 'APC', 'OTH']
    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    plot_cm(cm=cm, display_labels=display_labels)
    return correct / len(loader.dataset)


print("******Start inference on test set*****")
start_2 = time.time()
inference(test_loader)
end_2   = time.time()
print(f"Total inference time (min): {(end_2 - start_2) / 60:.2f}")
