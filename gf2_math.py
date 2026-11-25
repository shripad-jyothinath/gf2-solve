"""
gf2_math.py
===========
Galois Field GF(2) / F_2 Linear Algebra Engine for DRAM Reverse Engineering.

Mathematical Foundations:
- Physical addresses are vectors A in (F_2)^n.
- Bank selection functions are linear projections b = M * A (mod 2),
  where M is a k x n binary matrix of full row rank k.
- Two addresses A1, A2 map to the same bank iff:
      M * (A1 ^ A2) = 0 (mod 2)  <=>  (A1 ^ A2) in ker(M)
- The linear subspace ker(M) has dimension n - k.
- Its dual space (orthogonal complement) in (F_2)^n is the row space of M:
      Row(M) = (ker(M))^perp
- Two mapping matrices M1, M2 produce identical bank partitioning iff:
      RREF(M1) == RREF(M2)  over F_2.
"""

import numpy as np
from typing import Tuple, List, Optional

def dot_gf2(v1: np.ndarray, v2: np.ndarray) -> int:
    """Compute inner product over GF(2): sum(v1_i * v2_i) mod 2."""
    return int(np.bitwise_and(v1, v2).sum() % 2)

def matmul_gf2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two matrices/vectors over GF(2): (A @ B) % 2."""
    A_bin = (A % 2).astype(np.uint8)
    B_bin = (B % 2).astype(np.uint8)
    return (np.matmul(A_bin.astype(int), B_bin.astype(int)) % 2).astype(np.uint8)

def rref_gf2(matrix: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Compute the Row-Reduced Echelon Form (RREF) of a binary matrix over GF(2)
    using Gaussian elimination with XOR row operations.

    Returns:
        rref: The row-reduced echelon form matrix.
        pivot_cols: List of column indices containing leading 1s (pivots).
    """
    A = (matrix.copy() % 2).astype(np.uint8)
    num_rows, num_cols = A.shape
    pivot_cols = []
    current_row = 0

    for col in range(num_cols):
        if current_row >= num_rows:
            break

        # Step 1: Find a pivot row in this column
        pivot_row = None
        for r in range(current_row, num_rows):
            if A[r, col] == 1:
                pivot_row = r
                break

        if pivot_row is None:
            # No pivot in this column, move to next column
            continue

        # Step 2: Swap current_row with pivot_row
        if pivot_row != current_row:
            A[[current_row, pivot_row]] = A[[pivot_row, current_row]]

        pivot_cols.append(col)

        # Step 3: Eliminate 1s in all other rows of this column using XOR
        for r in range(num_rows):
            if r != current_row and A[r, col] == 1:
                A[r] = np.bitwise_xor(A[r], A[current_row])

        current_row += 1

    return A, pivot_cols

def rank_gf2(matrix: np.ndarray) -> int:
    """Compute matrix rank over GF(2)."""
    rref, pivot_cols = rref_gf2(matrix)
    return len(pivot_cols)

def nullspace_gf2(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the basis of the null space (kernel) of a matrix over GF(2):
        { x in (F_2)^n | A * x = 0 (mod 2) }

    Returns:
        null_basis: A matrix of shape (dim_null, n) where each row is a basis vector.
    """
    A, pivot_cols = rref_gf2(matrix)
    num_rows, num_cols = A.shape
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(num_cols) if c not in pivot_set]

    dim_null = len(free_cols)
    if dim_null == 0:
        return np.zeros((0, num_cols), dtype=np.uint8)

    # For each free variable, construct a null space basis vector
    null_basis = np.zeros((dim_null, num_cols), dtype=np.uint8)
    for i, free_col in enumerate(free_cols):
        null_basis[i, free_col] = 1
        for row_idx, p_col in enumerate(pivot_cols):
            if A[row_idx, free_col] == 1:
                null_basis[i, p_col] = 1

    return null_basis

def orthogonal_complement_gf2(vectors: np.ndarray) -> np.ndarray:
    """
    Given a set of vectors V in (F_2)^n (matrix where rows are vectors),
    find the basis of its orthogonal complement V^perp:
        { m in (F_2)^n | m . v = 0 (mod 2) for all v in V }

    Notice: V^perp is exactly the nullspace of V!
        V @ m = 0 (mod 2)  <=>  m in nullspace(V)
    """
    return nullspace_gf2(vectors)

def are_subspaces_equal_gf2(M1: np.ndarray, M2: np.ndarray) -> bool:
    """
    Test whether the row spaces of M1 and M2 are identical over GF(2):
        Row(M1) == Row(M2)
    This is true iff their non-zero RREF forms are identical.
    """
    rref1, pivots1 = rref_gf2(M1)
    rref2, pivots2 = rref_gf2(M2)

    # Filter out all-zero rows
    nonzero1 = rref1[~np.all(rref1 == 0, axis=1)]
    nonzero2 = rref2[~np.all(rref2 == 0, axis=1)]

    if nonzero1.shape != nonzero2.shape:
        return False
    return np.array_equal(nonzero1, nonzero2)

def address_to_bit_vector(addr: int, num_bits: int) -> np.ndarray:
    """Convert an integer address to a binary vector of length num_bits (LSB at index 0)."""
    return np.array([(addr >> i) & 1 for i in range(num_bits)], dtype=np.uint8)

def bit_vector_to_address(vec: np.ndarray) -> int:
    """Convert a binary vector to an integer address (LSB at index 0)."""
    addr = 0
    for i, bit in enumerate(vec):
        if bit:
            addr |= (1 << i)
    return addr

if __name__ == "__main__":
    # Unit tests for GF(2) linear algebra engine
    print("Testing GF(2) Linear Algebra Engine...")
    M_test = np.array([
        [1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0]
    ], dtype=np.uint8)

    rref_M, pivots = rref_gf2(M_test)
    print("RREF of M_test:\n", rref_M)
    print("Pivots:", pivots)
    assert rank_gf2(M_test) == 2

    # Null space of M_test
    null_M = nullspace_gf2(M_test)
    print(f"Null space dimension: {null_M.shape[0]} (expected 5 - 2 = 3)")
    print("Null space basis vectors:\n", null_M)

    # Verify that M_test @ null_v = 0 (mod 2) for all basis vectors
    prod = matmul_gf2(M_test, null_M.T)
    print("M_test @ null_M.T (mod 2):\n", prod)
    assert np.all(prod == 0), "Null space verification failed!"

    # Orthogonal complement of null space should recover row space of M_test!
    recovered_M = orthogonal_complement_gf2(null_M)
    print("Recovered Row(M) from null space:\n", recovered_M)
    assert are_subspaces_equal_gf2(M_test, recovered_M), "Subspace recovery failed!"

    print("\n[✓] All GF(2) algebraic unit tests passed successfully!")
