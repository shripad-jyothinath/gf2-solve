"""
benchmark_noise.py
==================
Systematic Noise Tolerance & Cross-Validation Benchmark for DRAM Mapping Recovery.

Evaluates Primitive 3 across varying noise floors (0% to 25% jitter / measurement corruption):
1. Subspace Recovery Accuracy (Exact RREF match over GF(2))
2. Held-Out Pairwise Collision Cross-Validation (Precision, Recall, F1-Score on 500 test pairs)
3. Generates publication-ready diagnostic curves in ./results/noise_tolerance_curves.png
"""

import os
import json
import time
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

from dram_geometry import DRAMGeometrySimulator
from clustering import cluster_addresses_by_bank, extract_candidate_nullspace_vectors
from recovery_engine import AddressMappingRecoveryEngine
from gf2_math import are_subspaces_equal_gf2, matmul_gf2, address_to_bit_vector

def evaluate_cross_validation(
    recovered_M: np.ndarray,
    sim: DRAMGeometrySimulator,
    num_test_pairs: int = 500,
    random_seed: int = 999
) -> Dict[str, float]:
    """
    Cross-validate recovered mapping on unseen held-out address pairs.
    Tests whether predicted bank collisions match true hardware collisions:
        Predicted Collision <=> recovered_M @ A1 == recovered_M @ A2 (mod 2)
    """
    rng = np.random.RandomState(random_seed)
    tp, fp, tn, fn = 0, 0, 0, 0

    for _ in range(num_test_pairs):
        a1 = int(rng.randint(0, 1 << sim.n) & ~0x3F)
        a2 = int(rng.randint(0, 1 << sim.n) & ~0x3F)

        # Ground truth
        true_collision = sim.is_true_bank_collision(a1, a2)

        # Predicted by recovered matrix
        v1 = address_to_bit_vector(a1, sim.n)
        v2 = address_to_bit_vector(a2, sim.n)
        b1_pred = matmul_gf2(recovered_M, v1)
        b2_pred = matmul_gf2(recovered_M, v2)
        pred_collision = np.array_equal(b1_pred, b2_pred)

        if pred_collision and true_collision:
            tp += 1
        elif pred_collision and not true_collision:
            fp += 1
        elif not pred_collision and true_collision:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / num_test_pairs

    return {
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1_score": f1 * 100.0,
        "test_accuracy": accuracy * 100.0,
        "num_test_pairs": num_test_pairs
    }

def main():
    os.makedirs("./results", exist_ok=True)
    print("=" * 80)
    print(" PROJECT 02: GF(2)-SOLVE — NOISE TOLERANCE & CROSS-VALIDATION BENCHMARK")
    print(" Reverse-Engineering Unknown Boolean DRAM Projections from Noisy Latencies")
    print("=" * 80)

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    num_probe_addrs = 96
    num_addr_bits = 24
    num_bank_bits = 4

    results = []
    table_rows = []

    for noise in noise_levels:
        t0 = time.time()
        # Initialize simulator with current noise floor
        sim = DRAMGeometrySimulator(
            num_addr_bits=num_addr_bits,
            num_bank_bits=num_bank_bits,
            noise_level=noise,
            random_seed=42
        )

        # Generate random 64B-aligned probe addresses
        rng = np.random.RandomState(42)
        probe_addrs = [int(rng.randint(0, 1 << num_addr_bits) & ~0x3F) for _ in range(num_probe_addrs)]

        # Step 1: Statistical clustering
        labels, _, thresh = cluster_addresses_by_bank(probe_addrs, sim, num_clusters=1 << num_bank_bits)
        diff_vecs = extract_candidate_nullspace_vectors(probe_addrs, labels, num_addr_bits)

        # Step 2: GF(2) basis recovery
        engine = AddressMappingRecoveryEngine(num_addr_bits=num_addr_bits, num_bank_bits=num_bank_bits)
        rec_M, diag = engine.recover_mapping(diff_vecs)

        # Step 3: Exact algebraic subspace check
        exact_match = are_subspaces_equal_gf2(sim.M_true, rec_M)

        # Step 4: Held-out cross-validation
        cv_stats = evaluate_cross_validation(rec_M, sim, num_test_pairs=500)
        elapsed = time.time() - t0

        entry = {
            "noise_level": noise * 100.0,
            "threshold_ns": thresh,
            "candidate_vectors": len(diff_vecs),
            "recovered_rank": diag["recovered_rank"],
            "exact_subspace_match": exact_match,
            "precision": cv_stats["precision"],
            "recall": cv_stats["recall"],
            "f1_score": cv_stats["f1_score"],
            "test_accuracy": cv_stats["test_accuracy"],
            "mean_consensus_score": diag["mean_consensus_score"],
            "elapsed_s": elapsed
        }
        results.append(entry)

        status_str = "EXACT MATCH (100%)" if exact_match else "Partial / Approximate"
        table_rows.append([
            f"{noise*100:.1f}%",
            f"{thresh:.1f} ns",
            diag["recovered_rank"],
            f"{cv_stats['precision']:.1f}%",
            f"{cv_stats['recall']:.1f}%",
            f"{cv_stats['f1_score']:.1f}%",
            f"{cv_stats['test_accuracy']:.1f}%",
            status_str,
            f"{elapsed:.2f}s"
        ])
        print(f"Noise {noise*100:4.1f}% -> F1: {cv_stats['f1_score']:5.1f}% | Exact Subspace: {exact_match} ({elapsed:.2f}s)")

    # Print summary table
    headers = ["Noise Floor", "Otsu Thresh", "Rank", "Precision", "Recall", "F1-Score", "Test Acc", "Algebraic Match", "Time"]
    print("\n" + tabulate(table_rows, headers=headers, tablefmt="github"))

    # Save results to JSON
    json_path = "./results/recovery_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Benchmark log saved to: {json_path}")

    # Generate Publication-Quality Visualization Plots
    plot_path = "./results/noise_tolerance_curves.png"
    noises = [r["noise_level"] for r in results]
    f1s = [r["f1_score"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    accuracies = [r["test_accuracy"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GF(2) DRAM Mapping Recovery — Noise Tolerance & Generalization Bounds", fontsize=14, fontweight="bold")

    # (1) F1-Score & Accuracy vs. Noise
    ax1.plot(noises, f1s, marker="o", color="crimson", linewidth=2, label="F1-Score (Bank Collision)")
    ax1.plot(noises, accuracies, marker="s", color="navy", linewidth=2, linestyle="--", label="Overall Accuracy")
    ax1.axhline(100.0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_title("Held-Out Generalization vs. Measurement Jitter", fontweight="bold")
    ax1.set_xlabel("Measurement Noise Level (% Corrupted Queries)")
    ax1.set_ylabel("Metric (%)")
    ax1.set_ylim(40, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left")

    # (2) Precision vs. Recall
    ax2.plot(noises, precisions, marker="^", color="darkgreen", linewidth=2, label="Precision")
    ax2.plot(noises, recalls, marker="v", color="darkorange", linewidth=2, label="Recall")
    ax2.set_title("Precision and Recall under Noise", fontweight="bold")
    ax2.set_xlabel("Measurement Noise Level (% Corrupted Queries)")
    ax2.set_ylabel("Rate (%)")
    ax2.set_ylim(40, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[+] Diagnostic plots saved to: {plot_path}")

if __name__ == "__main__":
    main()
