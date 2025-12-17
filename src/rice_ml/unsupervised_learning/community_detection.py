import numpy as np
from typing import Optional, Union, Sequence, Any, Tuple
from collections import Counter

class LabelPropagation:
    r"""
    Community Detection using the Label Propagation Algorithm (LPA).

    LPA is a fast, near-linear time algorithm for finding communities (clusters) 
    in a graph. It works by propagating labels through the network until each 
    node belongs to the community label most frequently represented among its neighbors.

    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of iterations to allow for label propagation.
    random_state : Optional[int], default=None
        Seed for the random number generator, used for initial label assignment
        and for tie-breaking during label updates.

    Attributes
    ----------
    labels_ : np.ndarray, shape (n_nodes,)
        The final community label assigned to each node (0 to K-1).
    n_communities_ : int
        The number of unique communities found.
    n_iter_ : int
        The number of iterations performed until convergence or max_iter.
    """

    def __init__(self, max_iter: int = 100, random_state: Optional[int] = None):
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_: Optional[np.ndarray] = None
        self.n_communities_: Optional[int] = None
        self.n_iter_: int = 0

    def fit(self, adjacency_matrix: Union[np.ndarray, Sequence[Sequence[Any]]]) -> "LabelPropagation":
        r"""
        Runs the Label Propagation Algorithm on the provided adjacency matrix.

        The adjacency matrix must be symmetric (for an undirected graph) and 
        can be binary or weighted.

        Parameters
        ----------
        adjacency_matrix : array_like, shape (n_nodes, n_nodes)
            The adjacency matrix representing the graph structure. Non-zero entries 
            indicate a link (and can be used as weights).

        Returns
        -------
        self : object

        Raises
        ------
        ValueError
            If the adjacency matrix is not square or if it contains negative values.
        """
        # Ensure the input is a square, non-negative NumPy array

        A = np.asarray(adjacency_matrix, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Adjacency matrix must be square (n_nodes, n_nodes).")
        if np.any(A < 0):
            raise ValueError("Adjacency matrix must not contain negative values.")

        n_nodes = A.shape[0]
        rng = np.random.default_rng(self.random_state)
        
        # 1. Initialization: Assign a unique label to each node (1, 2, ..., n)
        # We use a mutable list of labels during iteration for easy updates
        labels = np.arange(n_nodes)
        
        # Nodes are processed in a random order in each iteration
        node_order = np.arange(n_nodes)
        
        # 2. Iterative Propagation
        for i in range(self.max_iter):
            self.n_iter_ = i + 1
            
            # Shuffle the order in which nodes are processed (crucial for LPA)
            rng.shuffle(node_order)
            
            labels_changed = False
            
            for node_idx in node_order:
                # Find the neighbors and their connection strength (weights)
                # neighbors_indices are indices where A[node_idx, :] is non-zero
                # weights are the values A[node_idx, neighbors_indices]
                
                # Get indices and weights of connected neighbors
                neighbor_weights = A[node_idx, :]
                neighbor_indices = np.flatnonzero(neighbor_weights)
                
                if neighbor_indices.size == 0:
                    continue # Isolated node, label won't change

                # Get the labels and weights of the neighbors
                neighbor_labels = labels[neighbor_indices]
                weights = neighbor_weights[neighbor_indices]
                
                # Create a weighted frequency map of neighbor labels
                # Counter(labels) returns counts, but we need weighted sum
                
                # Use np.unique to find unique labels and sum their corresponding weights
                unique_labels, indices = np.unique(neighbor_labels, return_inverse=True)
                
                # Sum the weights for each unique label
                weighted_votes = np.bincount(indices, weights=weights)
                
                # Find the label(s) with the maximum total weight
                max_vote = np.max(weighted_votes)
                best_label_indices = np.flatnonzero(weighted_votes == max_vote)
                best_labels = unique_labels[best_label_indices]
                
                # Tie-breaking: If multiple labels have the max vote, choose one randomly
                if best_labels.size > 1:
                    new_label = rng.choice(best_labels)
                else:
                    new_label = best_labels[0]
                
                # Update the label if it changes
                if new_label != labels[node_idx]:
                    labels[node_idx] = new_label
                    labels_changed = True
                    
            # 3. Check for Convergence
            if not labels_changed:
                break

        # Finalize attributes
        self.labels_ = labels
        self.n_communities_ = np.unique(labels).size
        return self

    def predict(self, adjacency_matrix: Union[np.ndarray, Sequence[Sequence[Any]]]) -> np.ndarray:
        r"""
        Runs the Label Propagation Algorithm and returns the community labels.
        (Convenience wrapper for fit().labels_)
        """
        self.fit(adjacency_matrix)
        if self.labels_ is None:
             raise RuntimeError("Fit failed to produce labels.")
        return self.labels_
