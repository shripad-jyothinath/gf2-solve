"""
dram_geometry.py
================
Realistic DRAM Topology & Timing Side-Channel Simulator.

Modeled after JEDEC GDDR6 / DDR4 standards used in modern GPUs (AMD RDNA / NVIDIA):
- Physical Address width: n bits (e.g. 24 bits for a test aperture of 16 MB)
- Cache Line Offset: bits [0..5] (64 bytes)
- Bank Selection: k bits (e.g. k = 4 for 16 banks, or k = 5 for 32 banks)
  calculated via an unknown linear projection: b = M_true * A (mod 2)
- Row Address: remaining higher bits (bits [k+6 .. n-1])

Timing Characteristics:
- Different Banks: ~120 ns (Fast, parallel row buffers)
- Same Bank, Same Row: ~95 ns (Row-buffer Hit)
- Same Bank, Different Row: ~320 ns (Row-buffer Conflict / Collision: tRP + tRCD)
"""

import numpy as np
from typing import Tuple, Dict
from gf2_math import matmul_gf2, address_to_bit_vector

class DRAMGeometrySimulator:
    def __init__(
        self,
        num_addr_bits: int = 24,
        num_bank_bits: int = 4,
        noise_level: float = 0.05,
        random_seed: int = 42
    ):
        """
        Args:
            num_addr_bits: Total physical address bits under test (n).
            num_bank_bits: Number of bank index bits (k) -> 2^k banks.
            noise_level: Probability of measurement noise / jitter flips.
            random_seed: Seed for reproducibility.
        """
        self.n = num_addr_bits
        self.k = num_bank_bits
        self.num_banks = 1 << num_bank_bits
        self.noise_level = noise_level
        self.rng = np.random.RandomState(random_seed)

        # Construct realistic XOR mapping matrix M_true
        # Typical AMD/Intel scheme: lower bank bits [6..9] XORed with higher row bits
        self.M_true = self._generate_realistic_mapping()

    def _generate_realistic_mapping(self) -> np.ndarray:
        """
        Generates a realistic linear mapping matrix M_true in F_2^(k x n).
        Bits [0..5] are cache line offsets (never used for bank selection).
        Bits [6 .. 5+k] are direct bank bits, each XORed with 1 or 2 higher row bits.
        """
        M = np.zeros((self.k, self.n), dtype=np.uint8)
        base_bit = 6  # First bit after 64B cache line

        for i in range(self.k):
            # Direct bank bit
            M[i, base_bit + i] = 1
            # XOR with higher row bits to disperse spatial access patterns
            row_xor_1 = base_bit + self.k + 2 + (i * 2)
            row_xor_2 = base_bit + self.k + 7 + (i * 3)
            if row_xor_1 < self.n:
                M[i, row_xor_1] = 1
            if row_xor_2 < self.n:
                M[i, row_xor_2] = 1

        return M

    def get_bank(self, address: int) -> int:
        """Compute the true hardware bank index for a physical address."""
        vec = address_to_bit_vector(address, self.n)
        b_vec = matmul_gf2(self.M_true, vec)
        bank_idx = 0
        for i, bit in enumerate(b_vec):
            if bit:
                bank_idx |= (1 << i)
        return bank_idx

    def get_row(self, address: int) -> int:
        """Compute the DRAM row index (higher address bits)."""
        return address >> (6 + self.k)

    def measure_pairwise_latency(
        self,
        addr1: int,
        addr2: int,
        num_repetitions: int = 10
    ) -> float:
        """
        Simulate the physical timing side-channel of accessing addr1 then addr2.
        Returns the mean observed latency in nanoseconds.
        """
        bank1 = self.get_bank(addr1)
        bank2 = self.get_bank(addr2)
        row1 = self.get_row(addr1)
        row2 = self.get_row(addr2)

        is_conflict = (bank1 == bank2) and (row1 != row2)
        is_hit = (bank1 == bank2) and (row1 == row2)

        latencies = []
        for _ in range(num_repetitions):
            # Check for random hardware jitter / arbitration noise
            effective_conflict = is_conflict
            if self.rng.rand() < self.noise_level:
                # Corrupted measurement: flip collision state
                effective_conflict = not effective_conflict

            if effective_conflict:
                # Row-buffer conflict latency (tRP + tRCD + bus delay)
                lat = self.rng.normal(loc=320.0, scale=20.0)
            elif is_hit:
                # Row-buffer hit latency
                lat = self.rng.normal(loc=95.0, scale=8.0)
            else:
                # Different bank parallel access latency
                lat = self.rng.normal(loc=125.0, scale=12.0)

            latencies.append(max(lat, 50.0))

        return float(np.mean(latencies))

    def is_true_bank_collision(self, addr1: int, addr2: int) -> bool:
        """Ground truth oracle: True if addr1 and addr2 map to the same bank."""
        return self.get_bank(addr1) == self.get_bank(addr2)

    def describe(self) -> str:
        """Return a human-readable representation of the true XOR mapping equations."""
        lines = [f"DRAM Topology: {self.n}-bit Physical Address, {self.num_banks} Banks (k={self.k})"]
        lines.append("Ground Truth Hardware XOR Mapping Equations:")
        for i in range(self.k):
            active_bits = [f"A[{col}]" for col in range(self.n) if self.M_true[i, col] == 1]
            eq = " ^ ".join(active_bits)
            lines.append(f"  Bank Bit {i} (b_{i}) = {eq}")
        return "\n".join(lines)

if __name__ == "__main__":
    sim = DRAMGeometrySimulator(num_addr_bits=24, num_bank_bits=4, noise_level=0.05)
    print(sim.describe())
    print("\nTesting Latency Oracle:")
    a1 = 0x100040  # 64B aligned
    a2 = 0x100080  # adjacent line
    a3 = 0x200040  # different row
    print(f"Lat(a1, a2) [bank {sim.get_bank(a1)} vs {sim.get_bank(a2)}]: {sim.measure_pairwise_latency(a1, a2):.1f} ns")
    print(f"Lat(a1, a3) [bank {sim.get_bank(a1)} vs {sim.get_bank(a3)}]: {sim.measure_pairwise_latency(a1, a3):.1f} ns")
