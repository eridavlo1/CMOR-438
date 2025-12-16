# Community Detection: Label Propagation

This package implements the **Label Propagation Algorithm (LPA)**, a fast, near-linear time method for finding communities (or clusters) within a graph or network. LPA is a technique used in **Unsupervised Learning** to discover inherent group structures in data represented by an adjacency matrix.

## Algorithm Overview

LPA operates by propagating labels across a network until a stable community structure emerges.

### Core Mechanism

1. **Initialization:** Every node in the network is initially assigned a unique community label.
2. **Propagation:** The algorithm iterates, and in each step, every node updates its own label to the label that is **most frequently** represented among its immediate neighbors. If the graph is weighted, the node adopts the label with the **highest total weight** from its neighbors.
3. **Convergence:** The process stops when no node changes its label in a full iteration, or when the maximum number of iterations (`max_iter`) is reached.

### Key Characteristics

* **Fast:** The algorithm is highly efficient and scalable to very large networks.
* **No Prior Knowledge:** It does not require knowing the number of communities ($K$) beforehand.
* **Randomness:** The quality of the final communities can be sensitive to the initial random assignment and the random tie-breaking mechanism, which is controlled by `random_state`.

## Key Hyperparameters

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `max_iter` | `int` | **Maximum Iterations.** The hard limit on the number of times labels can be propagated. | `100` |
| `random_state` | `int` | Seed used for the initial random order of node processing and tie-breaking. | `None` |

---

## Data Requirements

LPA requires the input data to be structured as a graph adjacency matrix.

### Adjacency Matrix ($\mathbf{A}$)

* **Format:** Must be a square, 2D NumPy array where $A_{ij}$ represents the weight (strength) of the connection between node $i$ and node $j$.
* **Type:** Entries must be non-negative (binary or weighted).
* **Context:** The number of rows/columns is equal to the number of nodes in the network.
