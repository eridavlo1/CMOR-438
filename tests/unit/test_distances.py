import pytest
import numpy as np
from rice_ml import euclidean_distance, manhattan_distance

# --- 1. Basic Functionality and Precision ---

def test_distance_accuracy_and_precision(euclidean_distance, manhattan_distance):
    """Tests standard geometric cases and floating-point accuracy."""
    
    # Euclidean: 3-4-5 triangle
    assert np.allclose(euclidean_distance(np.array([0, 0]), np.array([3, 4])), 5.0)
    
    # Euclidean with negative numbers
    assert np.allclose(euclidean_distance([-1, -1], [2, 3]), 5.0) # sqrt(3^2 + 4^2)
    
    # Manhattan: |1-4| + |2-0| + |3-3| = 5.0
    assert np.allclose(manhattan_distance(np.array([1, 2, 3]), np.array([4, 0, 3])), 5.0)
    
    # Zero distance
    a = np.array([1.23456, 7.89])
    assert np.allclose(euclidean_distance(a, a), 0.0)
    assert np.allclose(manhattan_distance(a, a), 0.0)


# --- 2. Input Validation and Edge Cases ---

def test_shape_mismatch_validation(euclidean_distance, manhattan_distance):
    """Tests ValueError when input vector shapes do not match."""
    with pytest.raises(ValueError, match="same shape"):
        euclidean_distance(np.array([1, 2, 3]), np.array([1, 2]))
    with pytest.raises(ValueError, match="same shape"):
        manhattan_distance(np.array([1, 2]), np.array([1, 2, 3]))


def test_dimensionality_validation(euclidean_distance, manhattan_distance):
    """Tests ValueError for inputs that are not 1D vectors."""
    # Test 2D matrix input
    X = np.array([[1, 2], [3, 4]])
    Y = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="1-dimensional"):
        euclidean_distance(X, Y)
    with pytest.raises(ValueError, match="1-dimensional"):
        manhattan_distance(X, Y)


def test_empty_array_edge_case(euclidean_distance, manhattan_distance):
    """Tests the behavior with empty arrays (distance should be 0)."""
    empty_array = np.array([], dtype=float)
    assert np.allclose(euclidean_distance(empty_array, empty_array), 0.0)
    assert np.allclose(manhattan_distance(empty_array, empty_array), 0.0)


# --- 3. Metric Properties ---

def test_symmetry_and_nonnegative_properties(euclidean_distance, manhattan_distance):
    """Tests the core properties of distance metrics (symmetry, non-negativity, identity)."""
    a = np.array([1.5, -2.5, 3.0])
    b = np.array([4.0, 5.0, -6.0])
    
    # Symmetry: d(a, b) == d(b, a)
    assert np.allclose(euclidean_distance(a, b), euclidean_distance(b, a))
    assert np.allclose(manhattan_distance(a, b), manhattan_distance(b, a))
    
    # Non-negativity: d(a, b) >= 0
    assert euclidean_distance(a, b) >= 0.0
    assert manhattan_distance(a, b) >= 0.0
    
    # Identity of Indiscernibles: d(a, a) == 0
    assert np.allclose(euclidean_distance(a, a), 0.0)
    assert np.allclose(manhattan_distance(a, a), 0.0)


# --- 4. Input Data Types ---

def test_non_numeric_type_validation(euclidean_distance, manhattan_distance):
    """Tests that the functions raise TypeError for non-numeric content."""
    # Non-numeric elements
    with pytest.raises(TypeError, match="numeric"):
        euclidean_distance(np.array([1, 2]), np.array(["a", 4]))
    with pytest.raises(TypeError, match="numeric"):
        manhattan_distance([1, 2], ["x", "y"])

    # Test lists/tuples of strings
    with pytest.raises(TypeError, match="numeric"):
        euclidean_distance(["1", "2"], ["3", "4"])

    # Test empty string (should fail astype(float))
    with pytest.raises(TypeError, match="numeric"):
        manhattan_distance([""], [1])