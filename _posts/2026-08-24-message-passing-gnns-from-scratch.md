---
layout: post
title: "Message-Passing GNNs from Scratch: GCN and GAT in Pure PyTorch"
date: 2026-08-24
slug: message-passing-gnns-from-scratch
description: A theory-to-code introduction to graph convolution and graph attention without PyTorch Geometric.
tags: graph-neural-networks pytorch geometric-deep-learning
categories: graph-learning
related_posts: false
giscus_comments: false
pretty_table: true
toc:
  sidebar: left
  collapse: expanded
---

Graph libraries let us define a graph neural network in a few lines. That is convenient, but it can hide the main operation: features are gathered along edges, transformed into messages, and reduced at destination nodes. In this article I implement two standard architectures—GCN and GAT—using only PyTorch tensors.

The aim is not to replace optimized libraries. It is to make every graph operation visible. There is no `GCNConv`, `GATConv`, message-passing base class, or `torch_geometric` dependency. The complete training code and experiments are in the [companion repository](https://github.com/rizkisalaheddine/message-passing-gnns-from-scratch).

## Graph data

A graph is a pair $G=(V,E)$, where $V$ is a set of nodes and $E$ is a set of edges. Nodes represent entities and edges represent relationships. In a molecular graph, atoms are nodes and bonds are edges. In a citation network, papers are nodes and citations form directed edges.

A graph used for machine learning can also contain:

- node features $X\in\mathbb{R}^{N\times F}$, such as atom types or document embeddings;
- edge features $e_{ij}$, such as bond types, distances, or relationship categories;
- targets attached to nodes, edges, or the complete graph.

Graphs differ from images and sequences because they have neither a regular grid nor a fixed ordering. The numeric index assigned to each node is arbitrary. A graph layer should therefore share parameters across nodes and aggregate a neighborhood independently of the order in which its edges are stored.

### Sparse edge representation

A dense $N\times N$ adjacency matrix is unnecessary for the operations below. We store every directed edge $j\rightarrow i$ as one source index and one destination index:

```python
import torch

src = torch.tensor([0, 1, 2, 2, 3])
dst = torch.tensor([1, 0, 0, 3, 2])

x = torch.tensor([
    [1.0, 0.0],
    [0.8, 0.2],
    [0.0, 1.0],
    [0.1, 0.9],
])

# One source feature vector for each edge.
edge_features = x[src]  # shape: [num_edges, num_features]
```

Indexing with `src` moves us from node space to edge space. Aggregating with `dst` will move the messages back to node space.

## Message passing

The message-passing framework separates a graph layer into three operations. At layer $\ell$, an edge $j\rightarrow i$ first produces a message

$$
m_{j\rightarrow i}^{(\ell)} =
M^{(\ell)}\!\left(h_i^{(\ell)},h_j^{(\ell)},e_{ji}\right).
$$

Incoming messages are combined with a permutation-invariant aggregation:

$$
\bar m_i^{(\ell)} =
\underset{j\in\mathcal{N}(i)}{\operatorname{AGG}}
m_{j\rightarrow i}^{(\ell)}.
$$

Finally, an update function produces the next node state:

$$
h_i^{(\ell+1)} =
U^{(\ell)}\!\left(h_i^{(\ell)},\bar m_i^{(\ell)}\right).
$$

Sum, mean, and max are common aggregators. A minimal sum-based implementation is:

```python
def message_passing(x, src, dst, message_fn, update_fn):
    messages = message_fn(x[src], x[dst])

    aggregated = torch.zeros(
        x.size(0), messages.size(-1),
        dtype=messages.dtype,
        device=messages.device,
    )
    aggregated.index_add_(0, dst, messages)

    return update_fn(x, aggregated)
```

GCN and GAT both follow this pattern. Their main difference is the weight assigned to each incoming message.

## Graph Convolutional Networks

The GCN introduced by Kipf and Welling can be motivated as a localized approximation to spectral graph convolution. Its practical propagation rule begins by adding self-loops:

$$
\widetilde A=A+I,
\qquad
\widetilde D_{ii}=\sum_j\widetilde A_{ij}.
$$

One layer then computes

$$
H^{(\ell+1)}=\sigma\!\left(
\widetilde D^{-\frac12}\widetilde A\widetilde D^{-\frac12}
H^{(\ell)}W^{(\ell)}
\right).
$$

For a destination node $i$, the matrix equation is equivalent to

$$
h_i^{(\ell+1)}=\sigma\!\left(
\sum_{j\in\mathcal{N}(i)\cup\{i\}}
\frac{1}{\sqrt{\widetilde d_i\widetilde d_j}}
h_j^{(\ell)}W^{(\ell)}
\right).
$$

The coefficient $1/\sqrt{\widetilde d_i\widetilde d_j}$ depends only on graph structure. It prevents high-degree nodes from dominating the scale of the aggregation.

### Normalization in edge space

```python
def gcn_normalization(src, dst, num_nodes):
    nodes = torch.arange(num_nodes, device=src.device)
    src_hat = torch.cat([src, nodes])
    dst_hat = torch.cat([dst, nodes])

    degree = torch.zeros(num_nodes, device=src.device)
    degree.index_add_(
        0,
        dst_hat,
        torch.ones_like(dst_hat, dtype=torch.float),
    )

    norm = degree[src_hat].rsqrt() * degree[dst_hat].rsqrt()
    return src_hat, dst_hat, norm
```

### Transform and aggregate

```python
def gcn_layer(x, src, dst, weight, bias=None):
    src, dst, norm = gcn_normalization(src, dst, x.size(0))

    z = x @ weight
    messages = norm.unsqueeze(-1) * z[src]

    out = torch.zeros_like(z)
    out.index_add_(0, dst, messages)

    if bias is not None:
        out = out + bias
    return out
```

Stacking two layers lets a node use information from nodes up to two hops away. Repeated aggregation also tends to align nearby representations. At large depth, this may make nodes difficult to distinguish—a phenomenon usually called oversmoothing.

## Graph Attention Networks

GAT replaces GCN's fixed degree-based coefficients with feature-dependent attention. Each head first transforms every node:

$$
z_i=Wh_i.
$$

For an edge $j\rightarrow i$, the original GAT scoring function can be written as

$$
e_{ji}=\operatorname{LeakyReLU}\!\left(
a_{\mathrm{src}}^\top z_j+a_{\mathrm{dst}}^\top z_i
\right).
$$

```python
import torch.nn.functional as F

def attention_logits(x, src, dst, weight, a_src, a_dst):
    z = x @ weight
    scores = (
        (z[src] * a_src).sum(dim=-1)
        + (z[dst] * a_dst).sum(dim=-1)
    )
    scores = F.leaky_relu(scores, negative_slope=0.2)
    return scores, z
```

The scores are normalized only across edges entering the same destination node:

$$
\alpha_{ji}=
\frac{\exp(e_{ji})}
{\sum_{k\in\mathcal{N}(i)\cup\{i\}}\exp(e_{ki})}.
$$

The following version uses a loop for clarity. The repository also contains the reusable implementation used in the experiments.

```python
def neighbor_softmax(scores, dst, num_nodes):
    alpha = torch.empty_like(scores)
    for i in range(num_nodes):
        incoming = dst == i
        alpha[incoming] = torch.softmax(scores[incoming], dim=0)
    return alpha
```

The normalized coefficients weight the transformed source features:

$$
h_i'=\sigma\!\left(
\sum_{j\in\mathcal{N}(i)\cup\{i\}}
\alpha_{ji}z_j
\right).
$$

```python
def gat_head(x, src, dst, weight, a_src, a_dst):
    scores, z = attention_logits(x, src, dst, weight, a_src, a_dst)
    alpha = neighbor_softmax(scores, dst, x.size(0))

    messages = alpha.unsqueeze(-1) * z[src]
    out = torch.zeros_like(z)
    out.index_add_(0, dst, messages)
    return out, alpha
```

Several independently parameterized attention heads are usually combined. Hidden layers concatenate their outputs; a final prediction layer often averages them:

$$
h_i'=\mathbin\Vert_{k=1}^{K}h_i'^{(k)}
\quad\text{or}\quad
h_i'=\frac{1}{K}\sum_{k=1}^{K}h_i'^{(k)}.
$$

```python
head_outputs = [
    gat_head(x, src, dst, W[k], a_src[k], a_dst[k])[0]
    for k in range(num_heads)
]

hidden = torch.cat(head_outputs, dim=-1)
output = torch.stack(head_outputs).mean(0)
```

Attention makes aggregation adaptive, but an attention coefficient should not automatically be interpreted as a faithful explanation of a prediction. GAT also remains a local message-passing architecture and shares many of the depth limitations of GCN.

## GCN and GAT compared

| Aspect                | GCN                                        | GAT                                |
| --------------------- | ------------------------------------------ | ---------------------------------- |
| Neighbor weight       | Fixed degree normalization                 | Learned from endpoint features     |
| Parameters            | Feature transform $W$                      | $W$ and attention vectors per head |
| Aggregation           | Normalized weighted sum                    | Neighborhood-softmax weighted sum  |
| Inductive application | Yes, as a local propagation rule           | Yes                                |
| Main advantage        | Simple and parameter-efficient             | Adaptive neighborhood weighting    |
| Main cost             | Same structural weighting for all examples | Edge-level scores for every head   |

Both sparse implementations have work proportional to the number of represented edges. GAT additionally computes and stores a score for each edge and attention head. Whether that flexibility improves prediction depends on the data, model capacity, split, and optimization.

### Controlled synthetic experiments

The companion repository compares two-layer, hidden-width-16 models on three synthetic node-classification regimes and one graph-regression task. The table reports means over seeds 0, 1, and 2; full uncertainty values and the experiment configuration are kept with the code.

| Task                     |     Metric |       GCN |       GAT |
| ------------------------ | ---------: | --------: | --------: |
| High-homophily nodes     | Accuracy ↑ |     0.981 | **1.000** |
| Moderate-homophily nodes | Accuracy ↑ | **0.593** |     0.574 |
| Low-homophily nodes      | Accuracy ↑ | **0.333** |     0.315 |
| Graph regression         |      MAE ↓ | **0.162** |     0.171 |

Both models solve the strongly homophilous setting. As connectivity becomes less aligned with class labels, performance approaches the balanced three-class chance level of about $0.333$. GCN has slightly better means in the moderate- and low-homophily settings, but three synthetic seeds are not evidence of general superiority.

### Oversmoothing

A separate six-layer experiment measures mean pairwise cosine similarity and normalized node dispersion. Greater cosine similarity and lower dispersion indicate more aligned node representations.

| Model | Layer-6 cosine ↓ | Layer-6 dispersion ↑ | Accuracy ↑ |
| ----- | ---------------: | -------------------: | ---------: |
| GCN   |        **0.815** |            **0.265** |  **0.593** |
| GAT   |            0.934 |                0.072 |      0.481 |

In this particular single-head configuration, the GAT representations are more aligned by layer six. This result is configuration-specific; it does not imply that GAT always oversmooths more than GCN. The practical lesson is to measure representation collapse rather than assume that attention prevents it.

These datasets are synthetic, the test sets are small, and the implementations prioritize transparency over optimized kernels. The results are intended to make behavior reproducible, not to serve as a leaderboard.

## Takeaways

- A graph layer must be insensitive to arbitrary node indexing.
- Sparse gather and scatter operations form the computational core of message passing.
- GCN is a simple degree-normalized baseline.
- GAT learns local message weights but does not guarantee better accuracy or interpretability.
- Model depth should be evaluated together with oversmoothing diagnostics.

The full implementation, tests, experiment scripts, and reported results are available in [Message-Passing GNNs from Scratch](https://github.com/rizkisalaheddine/message-passing-gnns-from-scratch).

## References

1. Gilmer et al., [Neural Message Passing for Quantum Chemistry](https://proceedings.mlr.press/v70/gilmer17a.html), ICML 2017.
2. Kipf and Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), ICLR 2017.
3. Veličković et al., [Graph Attention Networks](https://arxiv.org/abs/1710.10903), ICLR 2018.
