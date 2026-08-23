---
layout: page
title: Message-Passing GNNs from Scratch
description: GCN and GAT implemented and compared in pure PyTorch.
importance: 1
category: research
---

This project implements the essential operations behind Graph Convolutional Networks and Graph Attention Networks using PyTorch tensors only. It deliberately avoids PyTorch Geometric so that gathering edge features, computing messages, and aggregating at destination nodes remain visible.

The accompanying experiments compare GCN and GAT on node classification, graph regression, and representation oversmoothing. The implementations favor readability and reproducibility over highly optimized graph kernels.

[Repository](https://github.com/rizkisalaheddine/message-passing-gnns-from-scratch) · [Read the tutorial](/blog/2026/message-passing-gnns-from-scratch/)
