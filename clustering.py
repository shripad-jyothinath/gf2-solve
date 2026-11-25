"""
clustering.py
=============
Noise-Resilient Statistical Filtering & Spectral Graph Clustering for DRAM Banks.

Problem Formulation:
Given N probe addresses, measuring all pairwise access latencies yields an
N x N noisy latency matrix W. Addresses targeting the same DRAM bank (but different rows)
exhibit elevated latencies due to row-buffer conflicts (tRP + tRCD).

Under an ideal oracle, the collision graph consists of 2^k disjoint cliques
(one for each DRAM bank). Under physical noise (bus contention, CPU jitter),
false edges (false positives) and missing edges (false negatives) perturb the graph.

Pipeline:
1. Bimodal Latency Thresholding (Otsu's / GMM threshold separation)
2. Normalized Graph Laplacian Construction: L_sym = I - D^(-1/2) * A * D^(-1/2)
3. Spectral Graph Clustering to partition N addresses into B = 2^k bank clusters
4. Extraction of candidate null-space difference vectors: Delta A = A_i ^ A_j
"""

import numpy as np
from sklearn.cluster import SpectralClustering
from typing import List, Tuple, Dict
from dram_geometry import DRAMGeometrySimulator

def determine_conflict_threshold(latency_matrix: np.ndarray) -> float:
    """
    Automatically determine the latency separation threshold between
    row-buffer hits/different-bank accesses (~120ns) and row-buffer conflicts (~320ns)
    using Otsu's method on the pairwise latency distribution.
    """
    # Extract upper triangular values (symmetric pairwise latencies)
    num_nodes = latency_matrix.shape[0]
    triu_indices = np.triu_indices(num_nodes, k=1)
    values = latency_matrix[triu_indices]

    # Otsu's thresholding over 100 histogram bins
    hist, bin_edges = np.histogram(values, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total_weight = values.size
    current_max, threshold = 0.0, float(np.mean(values))
    weight_background, sum_background = 0.0, 0.0
    sum_total = np.sum(hist * bin_centers)

    for i in range(len(hist)):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total_weight - weight_background
        if weight_foreground == 0:
            break

        sum_background += hist[i] * bin_centers[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        # Between-class variance
        var_between = weight_background * weight_foreground * ((mean_background - mean_foreground) ** 2)
        if var_between > current_max:
            current_max = var_between
            threshold = float(bin_centers[i])

    return threshold

def build_affinity_matrix(latency_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert raw latency measurements into a continuous affinity/adjacency matrix.
    Uses a sigmoid kernel centered at the conflict threshold to weigh edge confidence.
    """
    # Sigmoid scaling: values clearly above threshold -> near 1; below -> near 0
    scale = 15.0  # Transition slope in nanoseconds
    affinity = 1.0 / (1.0 + np.exp(-(latency_matrix - threshold) / scale))
    np.fill_diagonal(affinity, 0.0)
    return affinity

def cluster_addresses_by_bank(
        addresses: List[int],
        sim: DRAMGeometrySimulator,
        num_clusters: int,
        random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Measure pairwise latencies across addresses and cluster into candidate banks.

    Returns:
        cluster_labels: Assigned cluster index (0..num_clusters-1) for each address.
        latency_matrix: Raw pairwise N x N latency measurements.
        threshold: The detected row-conflict separation threshold.
    """
    num_addresses = len(addresses)
    latency_matrix = np.zeros((num_addresses, num_addresses), dtype=float)

    # Measure pairwise latencies (symmetric query matrix)
    for i in range(num_addresses):
        for j in range(i + 1, num_addresses):
            lat = sim.measure_pairwise_latency(addresses[i], addresses[j])
            latency_matrix[i, j] = lat
            latency_matrix[j, i] = lat

    # Compute optimal conflict threshold
    threshold = determine_conflict_threshold(latency_matrix)

    # Build continuous affinity matrix for spectral clustering
    affinity = build_affinity_matrix(latency_matrix, threshold)

    # Perform Spectral Clustering
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_seed,
        n_init=10
    )
    cluster_labels = clustering.fit_predict(affinity)

    return cluster_labels, latency_matrix, threshold

def extract_candidate_nullspace_vectors(
    addresses: List[int],
    cluster_labels: np.ndarray,
    num_addr_bits: int
) -> np.ndarray:
    """
    For all pairs of addresses assigned to the same cluster:
        v = A_i ^ A_j
    Under the model, v is a candidate vector in ker(M):
        M * v = 0 (mod 2)

    Returns:
        candidate_vectors: Matrix of shape (num_pairs, num_addr_bits) over GF(2).
    """
    from gf2_math import address_to_bit_vector

    candidate_vectors = []
    num_clusters = len(np.unique(cluster_labels))

    for c in range(num_clusters):
        cluster_indices = np.where(cluster_labels == c)[0]
        # Form all pairwise combinations within this cluster
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                idx1 = cluster_indices[i]
                idx2 = cluster_indices[j]
                diff_int = addresses[idx1] ^ addresses[idx2]
                vec = address_to_bit_vector(diff_int, num_addr_bits)
                candidate_vectors.append(vec)

    if not candidate_vectors:
        return np.zeros((0, num_addr_bits), dtype=np.uint8)

    return np.array(candidate_vectors, dtype=np.uint8)

if __name__ == "__main__":
    print("Testing Statistical Thresholding and Spectral Clustering...")
    sim = DRAMGeometrySimulator(num_addr_bits=24, num_bank_bits=4, noise_level=0.05)
    
    # Generate 128 random 64B-aligned probe addresses
    rng = np.random.RandomState(42)
    sample_addrs = [int(rng.randint(0, 1 << 24) & ~0x3F) for _ in range(64)]

    labels, lat_mat, thresh = cluster_addresses_by_bank(sample_addrs, sim, num_clusters=16)
    print(f"Detected Row-Conflict Threshold: {thresh:.1f} ns")
    print(f"Cluster label distribution across {len(sample_addrs)} addresses:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, cnt in zip(unique, counts):
        print(f"  Cluster {u:2d}: {cnt} addresses")

    cand_vecs = extract_candidate_nullspace_vectors(sample_addrs, labels, num_addr_bits=24)
    print(f"\nExtracted {cand_vecs.shape[0]} candidate null-space difference vectors.")
    print("[✓] Clustering module test completed.")
