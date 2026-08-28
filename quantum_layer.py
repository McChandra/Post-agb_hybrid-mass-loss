"""
quantum_layer.py
================
Quantum layer of the hybrid model.

Two complementary quantum formulations are provided:

1. QUBO (Quadratic Unconstrained Binary Optimisation)
   ------------------------------------------------
   Used for *discrete model selection / configuration optimisation* on top of the
   continuous Bayesian inference.  Example use: selecting the optimal combination of
   (disc class, wind on/off, inner-rim model, number of outflow components) that
   minimises a penalised chi^2.  Each binary variable encodes a modelling choice;
   the QUBO matrix Q is built from the chi^2 landscape evaluated on a coarse grid
   plus pairwise interaction penalties (e.g. "transition disc" is incompatible with
   "R_in = sublimation radius").

   The QUBO is solved exactly (brute force for small N), with a classical simulated
   annealer, and mapped to an Ising Hamiltonian for the VQE/QAOA path.

2. VQE (Variational Quantum Eigensolver)
   -------------------------------------
   Used to find the ground state of an *effective interaction Hamiltonian* that
   encodes the binary-disc coupling energy landscape.  We build a qubit Hamiltonian

       H = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j + sum_i g_i X_i

   whose coefficients (h_i, J_{ij}, g_i) are derived from the posterior means of the
   physical parameters (mass ratio q, eccentricity e, disc opening angle, ...).
   The ground-state energy / configuration then encodes the most probable
   large-scale geometry of the system (disc-dominated vs outflow-dominated, jet
   on/off, cavity size bin), providing a quantum-consistency cross-check on the
   classical posterior.

   Implemented with Qiskit (StatevectorSimulator / Aer) using a hardware-efficient
   ansatz and the COBYLA / SPSA optimiser.

Both pieces are deliberately *small* (<= 12 qubits) so they run exactly on a
statevector simulator in the sandbox and can be ported to real hardware.
"""

from __future__ import annotations

import itertools
import numpy as np

# ---------------------------------------------------------------------------
# QUBO
# ---------------------------------------------------------------------------
def build_qubo_from_chi2(chi2_grid: np.ndarray, penalty: float = 6.0,
                         incompat_pairs=None, groups=None):
    """
    Build a QUBO matrix Q (minimise x^T Q x, x in {0,1}^N) from a vector of
    chi^2 values for N candidate model configurations.

    To avoid the degenerate all-zero solution we add *one-hot* group
    constraints: for each group of mutually-exclusive choices we add the
    penalty  P * (sum_{i in group} x_i - 1)^2, which forces exactly one
    variable per group to be 1.  Expanded:
        P * [ sum_i x_i^2 - 2 sum_i x_i + 1 + 2 sum_{i<j} x_i x_j ]
    and since x_i^2 = x_i for binary vars, the linear term gets -P and the
    pairwise term gets +2P.

    chi2_grid : (N,) array of chi^2 for each single-choice configuration
    incompat_pairs : list of (i, j) index pairs that should not both be 1
    groups : list of lists of indices forming mutually-exclusive one-hot groups
    """
    N = len(chi2_grid)
    Q = np.zeros((N, N))
    # diagonal: linear coefficients = chi^2 (normalised)
    c = (chi2_grid - chi2_grid.min()) / (np.ptp(chi2_grid) + 1e-12)
    np.fill_diagonal(Q, c)
    # one-hot group constraints
    if groups:
        for grp in groups:
            for i in grp:
                Q[i, i] += -penalty          # linear part of P(x_i^2 - 2x_i)
            for ii in range(len(grp)):
                for jj in range(ii + 1, len(grp)):
                    i, j = grp[ii], grp[jj]
                    Q[i, j] += 2.0 * penalty / 2.0   # pairwise (symmetric)
                    Q[j, i] += 2.0 * penalty / 2.0
    # extra explicit incompatibility penalties
    if incompat_pairs:
        for (i, j) in incompat_pairs:
            Q[i, j] += penalty / 2.0
            Q[j, i] += penalty / 2.0
    return Q


def qubo_bruteforce(Q: np.ndarray):
    """Exact QUBO solution by enumeration (fine for N <= ~20)."""
    N = Q.shape[0]
    best_x, best_E = None, np.inf
    for bits in itertools.product([0, 1], repeat=N):
        x = np.array(bits, dtype=float)
        E = x @ Q @ x
        if E < best_E:
            best_E, best_x = E, x.copy()
    return best_x, best_E


def qubo_simulated_annealing(Q: np.ndarray, n_steps: int = 20000,
                             T0: float = 2.0, seed: int = 0):
    """Classical simulated annealing fallback for larger QUBO instances."""
    rng = np.random.default_rng(seed)
    N = Q.shape[0]
    x = rng.integers(0, 2, N).astype(float)
    E = x @ Q @ x
    best_x, best_E = x.copy(), E
    for k in range(n_steps):
        T = T0 * (1.0 - k / n_steps) + 1e-3
        i = rng.integers(N)
        x_new = x.copy(); x_new[i] = 1 - x_new[i]
        E_new = x_new @ Q @ x_new
        dE = E_new - E
        if dE < 0 or rng.random() < np.exp(-dE / T):
            x, E = x_new, E_new
            if E < best_E:
                best_E, best_x = E, x.copy()
    return best_x, best_E


def qubo_to_ising(Q: np.ndarray):
    """
    Map QUBO (x in {0,1}) to Ising (s in {-1,+1}) via x = (1+s)/2.
    Returns (J, h, offset) such that  E = s^T J s + h^T s + offset.
    """
    N = Q.shape[0]
    J = 0.25 * (Q + Q.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = 0.5 * np.diag(Q) + 0.25 * (Q.sum(axis=1) - np.diag(Q))
    offset = 0.5 * np.diag(Q).sum() + 0.25 * (Q.sum() - np.diag(Q).sum())
    return J, h, offset


# ---------------------------------------------------------------------------
# VQE
# ---------------------------------------------------------------------------
def build_interaction_hamiltonian(post_means: dict, n_qubits: int = 6):
    """
    Build an effective qubit Hamiltonian from posterior mean physical parameters.

    Encoding (n_qubits = 6 by default):
      q0 : disc-dominated (1) vs outflow-dominated (0)
      q1 : jet on (1) / off (0)
      q2 : transition disc (large cavity) (1) vs full disc (0)
      q3 : high eccentricity (1) vs low (0)
      q4 : high mass ratio (1) vs low (0)
      q5 : strong binary-disc coupling (1) vs weak (0)

    The local fields h_i and couplings J_ij are set by the posterior means so that
    the ground state of H reflects the most probable system configuration.
    """
    # Normalise posterior means to [0,1] "activation" of each qubit
    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    md_disc = post_means.get("log10_Mdot_disc", -6.5)
    md_wind = post_means.get("log10_Mdot_wind", -7.5)
    e = post_means.get("e", 0.2)
    q = post_means.get("q", 0.6)
    alpha = post_means.get("alpha_deg", 14.0)
    Rin = 10 ** post_means.get("log10_Rin_AU", np.log10(25.0))

    # local fields: negative h_i favours |1> (i.e. "active")
    h = np.zeros(n_qubits)
    h[0] = -sig((md_disc - md_wind) * 3.0)          # disc-dominated?
    h[1] = -sig((md_wind + 7.0) * 4.0)              # jet on?
    h[2] = -sig((Rin - 40.0) / 10.0)                # transition disc?
    h[3] = -sig((e - 0.25) * 8.0)                   # high e?
    h[4] = -sig((q - 0.6) * 6.0)                    # high q?
    h[5] = -sig((alpha - 14.0) / 4.0)               # strong coupling (thick disc)?

    # couplings: positive J_ij = antiferromagnetic (prefer different),
    #            negative J_ij = ferromagnetic (prefer same)
    J = np.zeros((n_qubits, n_qubits))
    J[0, 1] = -0.6   # disc-dominated tends to come with a jet (disc wind)
    J[0, 2] = 0.4    # disc-dominated vs transition disc: competing
    J[3, 5] = -0.5   # high e <-> strong coupling
    J[4, 5] = -0.3   # high q <-> strong coupling
    J[2, 5] = 0.3    # transition disc vs strong coupling: competing
    J[1, 5] = -0.4   # jet <-> strong coupling

    # transverse field (quantum fluctuation / tunnelling term)
    g = 0.3 * np.ones(n_qubits)

    return h, J, g


def hamiltonian_to_qubit_op(h, J, g):
    """Build a Qiskit SparsePauliOp from the Ising + transverse-field terms."""
    from qiskit.quantum_info import SparsePauliOp

    n = len(h)
    paulis, coeffs = [], []
    for i in range(n):
        if abs(h[i]) > 1e-12:
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            paulis.append("".join(label)); coeffs.append(h[i])
        if abs(g[i]) > 1e-12:
            label = ["I"] * n
            label[n - 1 - i] = "X"
            paulis.append("".join(label)); coeffs.append(g[i])
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                label = ["I"] * n
                label[n - 1 - i] = "Z"
                label[n - 1 - j] = "Z"
                paulis.append("".join(label)); coeffs.append(J[i, j])
    return SparsePauliOp(paulis, coeffs)


def run_vqe(h, J, g, reps: int = 2, maxiter: int = 200, seed: int = 0):
    """
    Run VQE with a hardware-efficient ansatz on a statevector simulator.
    Returns (ground_energy, optimal_params, statevector, ansatz).
    """
    from qiskit.circuit.library import EfficientSU2
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize

    op = hamiltonian_to_qubit_op(h, J, g)
    n = op.num_qubits
    ansatz = EfficientSU2(n, su2_gates=["ry", "rz"], entanglement="linear",
                          reps=reps, skip_final_rotation_layer=True)
    n_params = ansatz.num_parameters
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, n_params)

    def energy(p):
        qc = ansatz.assign_parameters(p)
        sv = Statevector.from_instruction(qc)
        return float(np.real(sv.expectation_value(op)))

    res = minimize(energy, x0, method="COBYLA",
                   options={"maxiter": maxiter, "rhobeg": 0.5})
    E0 = float(res.fun)
    sv = Statevector.from_instruction(ansatz.assign_parameters(res.x))
    return E0, res.x, sv, ansatz, op


def most_probable_bitstring(sv, n_qubits: int):
    """Return the most probable computational-basis bitstring of a statevector."""
    probs = np.abs(sv.data) ** 2
    idx = int(np.argmax(probs))
    return format(idx, f"0{n_qubits}b"), float(probs[idx])


# ---------------------------------------------------------------------------
# QAOA-style QUBO solver on the Ising-mapped Hamiltonian (bonus)
# ---------------------------------------------------------------------------
def solve_qubo_with_vqe(Q: np.ndarray, reps: int = 2, maxiter: int = 200,
                        seed: int = 0):
    """
    Map a QUBO to its Ising Hamiltonian and solve with VQE (ground state =
    optimal configuration).  Returns (best_bitstring, energy, probability).
    """
    J, h, offset = qubo_to_ising(Q)
    g = 0.1 * np.ones(len(h))  # small transverse field
    E0, params, sv, ansatz, op = run_vqe(h, J, g, reps=reps, maxiter=maxiter,
                                         seed=seed)
    n = len(h)
    bitstr, prob = most_probable_bitstring(sv, n)
    # bitstring s in {0,1}; convert Ising energy back (offset already excluded)
    return bitstr, E0 + offset, prob
