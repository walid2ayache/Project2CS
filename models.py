"""
GraphECGNet - Upgraded Architecture
====================================
Changes from the original:
  1. AttentiveSAGEConv  - SAGEConv enhanced with a lightweight attention gate
  2. KANActivation      - Learnable linear combination of basis functions (KAN-inspired)
  3. Dropout 0.5 -> 0.3 inside the GNN backbone
  4. GraphNorm          - Graph-aware normalisation between every layer
  5. Dual pooling       - global_mean_pool ⊕ global_max_pool (2× signal)
  6. Deeper head        - two-layer MLP classifier with residual normalisation
  7. Backward-compatible with main.py: still uses the same
     GraphGNNModel(c_in, c_out, layer_name, c_hidden, num_layers,
                   dp_rate_linear, dp_rate) signature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import (
    GATConv, GATv2Conv, GCNConv, GraphConv, SAGEConv,
    GraphNorm, global_mean_pool, global_max_pool, MessagePassing
)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  KAN-inspired activation
# ──────────────────────────────────────────────────────────────────────────────

class KANActivation(nn.Module):
    """
    Learnable activation that is a weighted sum of fixed basis functions.

    True KAN (Kolmogorov-Arnold Networks) uses learnable B-spline basis
    functions, which are expensive and have no native PyG support.  This
    module captures the core insight — the network learns *which shape* of
    nonlinearity to apply per channel — using a fixed, cheap basis:

        f(x) = w1·SiLU(x) + w2·tanh(x) + w3·sin(x) + w4·Gaussian(x)

    One set of four scalar weights is learned per channel (num_features),
    so the total extra parameter cost is 4 × c_hidden ≈ 256 params at
    c_hidden=64 — negligible.

    Args:
        num_features (int): number of input/output channels (must match
                            the previous layer's output width).
        init_std      (float): std of weight initialisation. Defaults to 0.1
                               so the activation starts close to a mix and
                               specialises during training.
    """

    def __init__(self, num_features: int, init_std: float = 0.1):
        super().__init__()
        # Four learnable scalars per channel — shape: (num_features, 4)
        self.weights = nn.Parameter(torch.randn(num_features, 4) * init_std)
        # Bias keeps the activation centred at init
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, num_features)
        w = F.softmax(self.weights, dim=-1)  # normalise so |weights|=1 initially

        b1 = F.silu(x)                              # smooth, gated
        b2 = torch.tanh(x)                          # bounded, saturating
        b3 = torch.sin(x)                           # periodic, for edge patterns
        b4 = torch.exp(-0.5 * x * x)               # Gaussian RBF, locality

        # w[:, k] is (num_features,) — broadcast over batch dimension N
        out = (w[:, 0] * b1 +
               w[:, 1] * b2 +
               w[:, 2] * b3 +
               w[:, 3] * b4 + self.bias)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Attentive SAGEConv
# ──────────────────────────────────────────────────────────────────────────────

class AttentiveSAGEConv(MessagePassing):
    """
    GraphSAGE aggregation enhanced with a lightweight attention gate.

    Standard SAGEConv:
        h_v = W1 · x_v  +  W2 · mean( x_u  for u in N(v) )

    This layer adds a scalar attention weight α(v, u) ∈ (0,1) before the mean:
        α(v, u)  = sigmoid( a · LeakyReLU( Wa · [x_v ∥ x_u] ) )
        h_v      = W1 · x_v  +  W2 · Σ_u α(v,u)·x_u / Σ_u α(v,u)

    The attention is cheap (one extra linear layer Wa: 2*c_in → 1) and lets the
    model learn to suppress noisy low-intensity background pixels even when they
    are geometric neighbours of an important edge pixel.

    Args:
        in_channels  (int): dimension of input node features
        out_channels (int): dimension of output node features
        bias         (bool): whether to add a bias term
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')   # we normalise manually → use 'add'
        self.in_channels  = in_channels
        self.out_channels = out_channels

        # Self projection
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        # Neighbour projection
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        # Attention scorer: maps [x_v ∥ x_u] -> scalar
        self.att = nn.Linear(2 * in_channels, 1, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.xavier_uniform_(self.att.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Propagate: fills self._alpha and accumulates weighted messages
        out = self.propagate(edge_index, x=x)          # (N, out_channels)
        out = self.lin_self(x) + out
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # x_i: (E, in_channels) — target node features (node v)
        # x_j: (E, in_channels) — source node features (neighbor u)
        pair = torch.cat([x_i, x_j], dim=-1)           # (E, 2*in_channels)
        alpha = torch.sigmoid(
            self.att(F.leaky_relu(pair, negative_slope=0.2))
        )                                               # (E, 1)
        # Store alpha so aggregate() can normalise
        self._alpha = alpha
        return alpha * self.lin_neigh(x_j)             # (E, out_channels)

    def aggregate(self, inputs: torch.Tensor,
                  index: torch.Tensor,
                  ptr=None,
                  dim_size=None) -> torch.Tensor:
        # Weighted sum of messages
        agg = super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        # Normalise by sum of attention weights so output is an attention-weighted mean
        alpha_sum = super().aggregate(
            self._alpha, index, ptr=ptr, dim_size=dim_size
        ).clamp(min=1e-6)
        return agg / alpha_sum


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Layer registry (original + new)
# ──────────────────────────────────────────────────────────────────────────────

gnn_layer_by_name = {
    "GCN"          : GCNConv,
    "GAT"          : GATConv,
    "GATv2"        : GATv2Conv,
    "GraphConv"    : GraphConv,
    "SAGE"         : SAGEConv,           # standard SAGEConv
    "AttentiveSAGE": AttentiveSAGEConv,  # ← new default recommendation
}


# ──────────────────────────────────────────────────────────────────────────────
# 4.  GNN backbone
# ──────────────────────────────────────────────────────────────────────────────

class GNNModel(nn.Module):
    """
    Stacked GNN backbone with:
      - Configurable layer type (default: AttentiveSAGE)
      - KAN activations between layers
      - GraphNorm for graph-size-invariant normalisation
      - Dropout at rate dp_rate (default 0.3, down from 0.5)

    Args:
        c_in       (int):   input feature dimension (1 for your grayscale pixel)
        c_hidden   (int):   hidden dimension
        c_out      (int):   output dimension (set to c_hidden in GraphGNNModel)
        num_layers (int):   total number of GNN layers
        layer_name (str):   key in gnn_layer_by_name
        dp_rate    (float): dropout inside the backbone
        **kwargs:           forwarded to the GNN layer constructor
    """

    def __init__(self,
                 c_in: int,
                 c_hidden: int,
                 c_out: int,
                 num_layers: int = 3,
                 layer_name: str = "AttentiveSAGE",
                 dp_rate: float = 0.3,
                 **kwargs):
        super().__init__()

        gnn_layer_cls = gnn_layer_by_name[layer_name]

        self.convs  = nn.ModuleList()
        self.norms  = nn.ModuleList()
        self.acts   = nn.ModuleList()
        self.drops  = nn.ModuleList()

        in_ch = c_in
        # All layers except the last project to c_hidden
        for idx in range(num_layers - 1):
            self.convs.append(
                gnn_layer_cls(in_channels=in_ch, out_channels=c_hidden, **kwargs)
            )
            self.norms.append(GraphNorm(c_hidden))
            self.acts.append(KANActivation(c_hidden))
            self.drops.append(nn.Dropout(dp_rate))
            in_ch = c_hidden

        # Final layer → c_out (= c_hidden in GraphGNNModel, actual classes elsewhere)
        self.convs.append(
            gnn_layer_cls(in_channels=in_ch, out_channels=c_out, **kwargs)
        )
        # No norm/act/drop after the last conv — pooling comes next

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          (Tensor): node features  (N_total, c_in)
            edge_index (Tensor): COO edge index (2, E)
            batch      (Tensor): batch vector   (N_total,)
        Returns:
            Tensor: node embeddings (N_total, c_out)
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x, batch)   # GraphNorm needs batch vector
            x = self.acts[i](x)
            x = self.drops[i](x)

        # Last conv — no post-processing here
        x = self.convs[-1](x, edge_index)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Full graph-classification model
# ──────────────────────────────────────────────────────────────────────────────

class GraphGNNModel(nn.Module):
    """
    Graph-level ECG arrhythmia classifier.

    Architecture:
        [Input: 1 grayscale feature per pixel-node]
          ↓
        GNNModel (AttentiveSAGE × num_layers, KAN activations, GraphNorm)
          ↓  node embeddings (N, c_hidden)
        global_mean_pool ⊕ global_max_pool  →  graph embedding (B, 2*c_hidden)
          ↓
        LayerNorm → Dropout(dp_rate_linear) → Linear(2*c_hidden, c_hidden)
          ↓
        KANActivation → Dropout(dp_rate_linear/2) → Linear(c_hidden, c_out)
          ↓
        [Output: class logits (B, c_out)]

    The dual pooling doubles the graph-level signal at zero training cost:
      - mean pool  → "average edge density" (texture / rhythm)
      - max pool   → "most active edge pixel" (peak signal presence)

    Args:
        c_in           (int):   node feature dimension (usually 1)
        c_out          (int):   number of classes (8 for MIT-BIH)
        c_hidden       (int):   hidden dimension (default 64)
        num_layers     (int):   GNN depth (default 3)
        layer_name     (str):   GNN layer type (default "AttentiveSAGE")
        dp_rate        (float): dropout inside GNN backbone (default 0.3)
        dp_rate_linear (float): dropout in the head (default 0.5, kept from
                                original for the head — you can lower to 0.3)
    """

    def __init__(self,
                 c_in: int,
                 c_out: int,
                 c_hidden: int = 64,
                 num_layers: int = 3,
                 layer_name: str = "AttentiveSAGE",
                 dp_rate: float = 0.3,
                 dp_rate_linear: float = 0.5,
                 **kwargs):
        super().__init__()

        self.gnn = GNNModel(
            c_in=c_in,
            c_hidden=c_hidden,
            c_out=c_hidden,      # backbone outputs c_hidden; head handles class projection
            num_layers=num_layers,
            layer_name=layer_name,
            dp_rate=dp_rate,
            **kwargs
        )

        # Dual-pool output is 2*c_hidden
        pool_out = 2 * c_hidden

        # Two-layer MLP head with KAN activation in the middle
        self.head = nn.Sequential(
            nn.LayerNorm(pool_out),
            nn.Dropout(dp_rate_linear),
            nn.Linear(pool_out, c_hidden),
            KANActivation(c_hidden),
            nn.Dropout(dp_rate_linear / 2),
            nn.Linear(c_hidden, c_out),
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          (Tensor): node features  (N_total, c_in)
            edge_index (Tensor): COO edge index (2, E)
            batch_idx  (Tensor): batch assignment (N_total,)
        Returns:
            Tensor: class logits (B, c_out) — no softmax applied
        """
        # Node-level embeddings
        node_emb = self.gnn(x, edge_index, batch_idx)   # (N_total, c_hidden)

        # Dual graph-level pooling
        g_mean = global_mean_pool(node_emb, batch_idx)  # (B, c_hidden)
        g_max  = global_max_pool(node_emb, batch_idx)   # (B, c_hidden)
        g      = torch.cat([g_mean, g_max], dim=-1)     # (B, 2*c_hidden)

        # Classification head
        return self.head(g)                              # (B, c_out)
