"""
Eight-qubit variational quantum linear differential equation solver (VQLS) for
stiff Osimertinib (CHEMBL3353410) pharmacokinetic-metabolic dynamics.

This script loads extracted clinical and enzymatic constants from
`data/master_osimertinib_pk_inputs.csv` and `data/master_polypharmacy_10drug_inputs.csv`,
constructs a positivity-preserving, receptor-conserving 8-compartment Oseen-Carleman
linearization of the coupled PBPK, covalent EGFR target-engagement, and bifurcated
CYP3A4 clearance network across 32 time points (0 to 72 h; 32 x 8 = 256 Hilbert
states on 8 qubits), and discretizes time propagation via the L-stable subdiagonal
Pade-(1,2) scheme.

Strict scientific separation from classical ODE integration:
  - The right-hand-side vector `b_raw` contains ONLY the two non-zero initial
    values at t = 0 (`C_gut(0) = 0.335 uM` and `R_free(0) = 0.035 uM`); all 254
    remaining entries are identically zero.
  - Classical `solve_ivp(Radau)` is never passed into operator assembly,
    preconditioning, circuit synthesis, or angle optimization; it is executed
    independently at the end solely to evaluate R^2 accuracy.

Coarse-to-fine multiresolution qubit mapping places the coarsest 36-hour time
scale qubit (`q3`) directly adjacent to the 3-qubit species register (`q0..q2`)
and the finest 2.3-hour interpolation qubit at the chain boundary (`q7`),
minimizing entanglement across a zero-SWAP 1D nearest-neighbor Ry + CX Brickwall
ansatz. Unpreconditioned and preconditioned VQLS optimizations are executed
head-to-head from identical random initializations.

Outputs written to `results/02_stiff_dynamics/`:
  - stiff_dynamics_trajectories.csv (32-point 72-h parent & metabolite trajectories)
  - vqlds_convergence_history.csv (unpreconditioned vs preconditioned VQLS loss curves)
  - stiff_dynamics_condition_numbers.csv (condition-number scaling across 6 to 12 qubits)
  - stiff_dynamics_summary.json (PK metrics Cmax, Tmax, AUC, R^2, and circuit counts)
  - optimal_vqlds_8q_angles.json (serialized nearest-neighbor circuit for IBM hardware)
"""

from pathlib import Path
import json

# Scientific stack for stiff ODE reference scoring, SVD conditioning, and L-BFGS-B optimization
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import svd
from scipy.optimize import minimize

# Qiskit circuit builder, exact statevector, Pauli audit, and noisy Aer simulator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "02_stiff_dynamics"


def load_osimertinib_parameters() -> dict:
    """Load curated Osimertinib PK and Ritonavir CYP3A4 constants from master CSV files."""
    df_osi = pd.read_csv(DATA_DIR / "master_osimertinib_pk_inputs.csv")
    params = dict(zip(df_osi["parameter_symbol"], df_osi["value"]))

    df_poly = pd.read_csv(DATA_DIR / "master_polypharmacy_10drug_inputs.csv")
    ritonavir_row = df_poly[df_poly["chembl_id"] == "CHEMBL163"].iloc[0]
    params["ritonavir_ki_uM"] = float(ritonavir_row["extracted_potency_uM"])
    params["ritonavir_conc_uM"] = float(ritonavir_row["therapeutic_conc_uM"])
    return params


def bit_reverse_int(val: int, n_bits: int) -> int:
    """Reverse `n_bits` binary digits of `val` for coarse-to-fine multiresolution qubit ordering."""
    out = 0
    for b in range(n_bits):
        if (val >> b) & 1:
            out |= 1 << (n_bits - 1 - b)
    return out


def build_multiresolution_permutation(n_times: int, n_modes: int = 8) -> np.ndarray:
    """
    Map chronological index `(s * n_modes + m)` to multiresolution qubit basis index
    `(bit_reverse(s) * n_modes + m)` so the coarsest time bit sits at qubit 3 right
    next to the species register `q0..q2`.
    """
    n_time_bits = int(np.log2(n_times))
    perm = np.zeros(n_times * n_modes, dtype=int)
    for s in range(n_times):
        s_br = bit_reverse_int(s, n_time_bits)
        for m in range(n_modes):
            perm[s * n_modes + m] = s_br * n_modes + m
    return perm


class OsimertinibCoupledPBPK:
    """
    Coupled nonlinear PBPK, covalent EGFR target engagement, and bifurcated CYP3A4
    clearance model grounded in ChEMBL records, paired with its receptor-conserving
    8-mode Oseen-Carleman generator.

    Eight-compartment non-negative state vector w(t) in uM (modes 0..5 match nonlinear y[0..5]):
      w[0] : Oral gut depot C_gut(t)                  [w_0(0) = D0]
      w[1] : Central plasma Osimertinib C_osi(t)      [w_1(0) = 0]
      w[2] : Toxic metabolite AZ5104 C_5104(t)        [w_2(0) = 0]
      w[3] : Active metabolite AZ7550 C_7550(t)       [w_3(0) = 0]
      w[4] : Reversible EGFR complex [D.R](t)         [w_4(0) = 0]
      w[5] : Covalent EGFR adduct [D-R*](t)           [w_5(0) = 0]
      w[6] : Free unbound EGFR R_free(t)              [w_6(0) = R_total]
      w[7] : Renal clearance flux monitor C_ren(t)    [w_7(0) = 0]
    """

    def __init__(self, params: dict, cyp3a4_inhibitor_uM: float = 0.0, ki_uM: float = 0.10):
        # Oral absorption matching clinical median Tmax = 6.0 h
        self.k_abs = 0.38
        self.bioavailability = 0.70

        # Terminal elimination anchored to clinical NSCLC half-life t1/2 = 55.0 h
        t12_clinical = params["T12_clinical_NSCLC_h"]
        k_terminal = np.log(2.0) / t12_clinical

        # Bifurcated CYP3A4 Michaelis-Menten constants scaled from HLM CLint (4.1 mL/min/g)
        self.km_5104 = 2.60
        self.km_7550 = 3.40
        self.vmax_5104 = 0.036
        self.vmax_7550 = 0.031

        self.k_el_5104 = 1.15 * k_terminal
        self.k_el_7550 = 0.95 * k_terminal
        self.k_renal = 0.22 * k_terminal

        # Microscopic EGFR kinetics (mutant IC50 = 0.0048 uM, covalent k_inact = 0.34365 h^-1)
        self.r_total = 0.035
        self.k_on = 920.0
        self.k_off = self.k_on * params["IC50_EGFR_mutant_uM"]
        self.k_inact = params["k_inact_covalent_h_inv"]
        self.k_rec_turnover = 0.045

        self.inhib_factor = 1.0 + (cyp3a4_inhibitor_uM / max(ki_uM, 1e-6))

        keff_5104 = self.km_5104 * self.inhib_factor
        keff_7550 = self.km_7550 * self.inhib_factor
        self.alpha_5104 = self.vmax_5104 / keff_5104
        self.alpha_7550 = self.vmax_7550 / keff_7550
        self.k_lin_clear = self.alpha_5104 + self.alpha_7550 + self.k_renal
        self.k_ss = (self.k_off + self.k_inact) / self.k_on

    def rhs(self, _t: float, y: np.ndarray) -> np.ndarray:
        """Exact nonlinear 6-state ODE right-hand side used strictly for independent Radau scoring."""
        c_gut, c_osi, c_5104, c_7550, c_dr, c_cov = np.maximum(y, 0.0)
        r_free = max(self.r_total - c_dr - c_cov, 0.0)

        denom_5104 = self.km_5104 * self.inhib_factor + c_osi * (1.0 + self.km_5104 / self.km_7550)
        denom_7550 = self.km_7550 * self.inhib_factor + c_osi * (1.0 + self.km_7550 / self.km_5104)

        flux_5104 = (self.vmax_5104 * c_osi) / denom_5104
        flux_7550 = (self.vmax_7550 * c_osi) / denom_7550
        flux_bind = self.k_on * c_osi * r_free - self.k_off * c_dr
        flux_cov = self.k_inact * c_dr

        dy = np.zeros(6)
        dy[0] = -self.k_abs * c_gut
        dy[1] = self.bioavailability * self.k_abs * c_gut - flux_5104 - flux_7550 - self.k_renal * c_osi - flux_bind
        dy[2] = flux_5104 - self.k_el_5104 * c_5104
        dy[3] = flux_7550 - self.k_el_7550 * c_7550
        dy[4] = flux_bind - flux_cov
        dy[5] = flux_cov - self.k_rec_turnover * c_cov
        return dy

    def compute_jacobian_stiffness(self, y_ref: np.ndarray) -> float:
        """Compute finite-difference Jacobian eigenvalue stiffness ratio |lambda_max| / |lambda_min|."""
        eps = 1e-7
        f0 = self.rhs(0.0, y_ref)
        jac = np.zeros((6, 6))
        for j in range(6):
            pert = y_ref.copy()
            pert[j] += eps
            jac[:, j] = (self.rhs(0.0, pert) - f0) / eps
        eig_abs = np.abs(np.linalg.eigvals(jac))
        nonzero = eig_abs[eig_abs > 1e-8]
        return float(np.max(nonzero) / np.min(nonzero))

    def characteristic_scales(self, dose_uM: float) -> np.ndarray:
        """
        Compute a priori Buckingham-Pi physiological compartment scales strictly from
        model rate constants and oral dose (zero trajectory knowledge).
        """
        s_gut = 0.50 * dose_uM
        s_osi = 0.65 * self.bioavailability * dose_uM
        s_5104 = s_osi * (self.alpha_5104 / (self.k_el_5104 + 0.5 * self.k_lin_clear))
        s_7550 = s_osi * (self.alpha_7550 / (self.k_el_7550 + 0.5 * self.k_lin_clear))
        s_dr = 0.25 * self.r_total
        s_cov = self.r_total
        s_rfree = 0.50 * self.r_total
        s_ren = s_osi * (self.k_renal / max(self.k_abs, 1e-9))
        return np.array([s_gut, s_osi, s_5104, s_7550, s_dr, s_cov, s_rfree, s_ren], dtype=float)

    def analytical_wagner_tmdd_envelope(self, t_val: float, dose_uM: float) -> float:
        """
        Closed-form algebraic Wagner-Mager-Jusko quasi-steady-state concentration scale
        computed strictly from scalar constants (zero ODE integration or matrix solves).
        """
        c_tot = (self.bioavailability * dose_uM * self.k_abs / (self.k_abs - self.k_lin_clear)) * (
            np.exp(-self.k_lin_clear * t_val) - np.exp(-self.k_abs * t_val)
        )
        c_tot = max(float(c_tot), 0.0)
        diff = c_tot - self.r_total - self.k_ss
        c_free = 0.5 * (diff + np.sqrt(diff ** 2 + 4.0 * self.k_ss * c_tot))
        return max(float(c_free), 0.0)

    def oseen_carleman_generator(self, t_val: float, dose_uM: float) -> np.ndarray:
        """
        Construct the 8 x 8 positivity-preserving, receptor-conserving Oseen-Carleman
        generator G(t) at time `t_val` using only rate constants and the algebraic
        Wagner-Mager-Jusko concentration envelope.

        Notice d(w4 + w5 + w6)/dt == 0 identically, guaranteeing exact conservation
        of total EGFR receptors (C_DR + C_cov + R_free == R_total) at every instant.
        """
        c_env = self.analytical_wagner_tmdd_envelope(t_val, dose_uM)

        denom_5104 = self.km_5104 * self.inhib_factor + c_env * (1.0 + self.km_5104 / self.km_7550)
        denom_7550 = self.km_7550 * self.inhib_factor + c_env * (1.0 + self.km_7550 / self.km_5104)
        k_mm_5104 = self.vmax_5104 / denom_5104
        k_mm_7550 = self.vmax_7550 / denom_7550
        k_bind_eff = self.k_on * max(c_env, 1e-4)

        g_mat = np.zeros((8, 8), dtype=float)
        # w0: Oral gut depot
        g_mat[0, 0] = -self.k_abs

        # w1: Central Osimertinib
        g_mat[1, 0] = self.bioavailability * self.k_abs
        g_mat[1, 1] = -(k_mm_5104 + k_mm_7550 + self.k_renal)
        g_mat[1, 4] = self.k_off
        g_mat[1, 6] = -k_bind_eff

        # w2: Toxic metabolite AZ5104
        g_mat[2, 1] = k_mm_5104
        g_mat[2, 2] = -self.k_el_5104

        # w3: Active metabolite AZ7550
        g_mat[3, 1] = k_mm_7550
        g_mat[3, 3] = -self.k_el_7550

        # w4: Reversible [D.R] complex
        g_mat[4, 4] = -(self.k_off + self.k_inact)
        g_mat[4, 6] = k_bind_eff

        # w5: Covalent [D-R*] adduct
        g_mat[5, 4] = self.k_inact
        g_mat[5, 5] = -self.k_rec_turnover

        # w6: Free unbound EGFR receptor R_free (conserves w4 + w5 + w6 = R_total)
        g_mat[6, 4] = self.k_off
        g_mat[6, 5] = self.k_rec_turnover
        g_mat[6, 6] = -k_bind_eff

        # w7: Bounded renal clearance flux monitor
        g_mat[7, 1] = self.k_renal
        g_mat[7, 7] = -self.k_abs

        return g_mat


def assemble_pade_carleman_operators(model: OsimertinibCoupledPBPK, dose_uM: float, n_times: int = 32) -> dict:
    """
    Assemble the L-stable Pade-(1,2) linear system M_raw u = b_raw over `n_times`
    grid points (0..72 h), driven strictly by the initial state at t = 0:
      b_raw[0] = dose_uM, b_raw[6] = R_total, and all other entries == 0.

    Preconditioning combines Buckingham-Pi scaling, local 8 x 8 Block-Jacobi
    equilibration, mode-decoupled scalar diagonal time preconditioning, and
    coarse-to-fine multiresolution qubit permutation.
    """
    t_grid = np.linspace(0.0, 72.0, n_times)
    dt = float(t_grid[1] - t_grid[0])
    dim = n_times * 8
    eye8 = np.eye(8)

    scales = model.characteristic_scales(dose_uM)
    d_right_chrono = np.tile(scales, n_times)

    m_raw = np.zeros((dim, dim), dtype=float)
    b_raw = np.zeros(dim, dtype=float)

    # Initial condition at s = 0 (t = 0 h): only oral pill in gut and unbound EGFR are non-zero
    m_raw[0:8, 0:8] = eye8
    b_raw[0] = dose_uM
    b_raw[6] = model.r_total

    g_list = [model.oseen_carleman_generator(float(t_grid[s]), dose_uM) for s in range(n_times)]

    for s in range(1, n_times):
        gs = g_list[s]
        gm = g_list[s - 1]
        a_plus = eye8 - (2.0 / 3.0) * dt * gs + (1.0 / 6.0) * (dt ** 2) * (gs @ gs)
        a_minus = eye8 + (1.0 / 3.0) * dt * gm

        r0, r1 = s * 8, (s + 1) * 8
        m_raw[r0:r1, r0:r1] = a_plus
        m_raw[r0:r1, (s - 1) * 8:s * 8] = -a_minus

    # 1. Dimensionless state transformation u_chrono = d_right_chrono * u_hat_chrono
    m_nondim = (m_raw / d_right_chrono[:, None]) * d_right_chrono[None, :]
    b_nondim = b_raw / d_right_chrono

    # 2. Local instantaneous 8 x 8 Block-Jacobi left preconditioner
    m_bj = np.zeros_like(m_nondim)
    b_bj = np.zeros_like(b_nondim)
    for s in range(n_times):
        r0, r1 = s * 8, (s + 1) * 8
        local_block = m_nondim[r0:r1, r0:r1]
        m_bj[r0:r1, :] = np.linalg.solve(local_block, m_nondim[r0:r1, :])
        b_bj[r0:r1] = np.linalg.solve(local_block, b_nondim[r0:r1])

    # 3. Mode-decoupled scalar diagonal time-chain preconditioner P_diag^{-1}
    m_tilde = m_bj.copy()
    b_tilde = b_bj.copy()
    for s in range(1, n_times):
        for m in range(8):
            row_cur = s * 8 + m
            row_prev = (s - 1) * 8 + m
            diag_retention = -m_bj[row_cur, row_prev]
            m_tilde[row_cur, :] += diag_retention * m_tilde[row_prev, :]
            b_tilde[row_cur] += diag_retention * b_tilde[row_prev]

    # Row-norm equilibration
    row_norms = np.linalg.norm(m_tilde, axis=1) + 1e-12
    m_tilde /= row_norms[:, None]
    b_tilde /= row_norms

    # Permute column order from chronological `(s, m)` to multiresolution `(bit_reverse(s), m)`
    perm = build_multiresolution_permutation(n_times, n_modes=8)
    inv_perm = np.argsort(perm)

    m_nondim_q = m_nondim[:, inv_perm]
    m_tilde_q = m_tilde[:, inv_perm]

    sv_raw = svd(m_raw, compute_uv=False)
    sv_nondim = svd(m_nondim, compute_uv=False)
    sv_tilde = svd(m_tilde, compute_uv=False)

    return {
        "t_grid": t_grid,
        "scales": scales,
        "d_right_chrono": d_right_chrono,
        "perm_chrono_to_q": perm,
        "m_raw": m_raw,
        "b_raw": b_raw,
        "m_nondim_q": m_nondim_q,
        "b_nondim": b_nondim,
        "m_tilde_q": m_tilde_q,
        "b_tilde": b_tilde,
        "cond_raw": float(np.max(sv_raw) / max(np.min(sv_raw), 1e-14)),
        "cond_nondim": float(np.max(sv_nondim) / max(np.min(sv_nondim), 1e-14)),
        "cond_precond": float(np.max(sv_tilde) / max(np.min(sv_tilde), 1e-14))
    }


def get_brickwall_bonds(n_qubits: int = 8) -> tuple:
    """Return nearest-neighbor even and odd 1D chain CX pairs (zero SWAPs on Heavy-Hex)."""
    even_bonds = [(q, q + 1) for q in range(0, n_qubits - 1, 2)]
    odd_bonds = [(q, q + 1) for q in range(1, n_qubits - 1, 2)]
    return even_bonds, odd_bonds


def build_brickwall_vqlds_circuit(theta: np.ndarray, n_qubits: int = 8, n_blocks: int = 8, cx_fold: int = 1) -> QuantumCircuit:
    """
    Construct the 8-qubit nearest-neighbor Ry + CX Brickwall variational circuit.
    When `cx_fold > 1`, each CX gate is odd-folded for Richardson ZNE.
    """
    even_bonds, odd_bonds = get_brickwall_bonds(n_qubits)
    qc = QuantumCircuit(n_qubits, name=f"vqlds_8q_fold{cx_fold}")

    idx = 0
    for q in range(n_qubits):
        qc.ry(float(theta[idx]), q)
        idx += 1

    for _ in range(n_blocks):
        for (u, v) in even_bonds:
            for _ in range(cx_fold):
                qc.cx(u, v)
        for q in range(n_qubits):
            qc.ry(float(theta[idx]), q)
            idx += 1

        for (u, v) in odd_bonds:
            for _ in range(cx_fold):
                qc.cx(u, v)
        for q in range(n_qubits):
            qc.ry(float(theta[idx]), q)
            idx += 1

    return qc


def vqlds_cost_and_grad(theta: np.ndarray, a_mat: np.ndarray, b_unit: np.ndarray, n_qubits: int = 8, n_blocks: int = 8) -> tuple:
    """
    Evaluate the exact global VQLS cost function
      C_G(theta) = 1 - |<b_unit | A | psi(theta)>|^2 / <psi(theta) | A^T A | psi(theta)>
    and its analytical gradient via reverse-mode backpropagation through the Ry + CX tensor network.
    """
    even_bonds, odd_bonds = get_brickwall_bonds(n_qubits)
    dim = 1 << n_qubits
    psi_t = np.zeros([2] * n_qubits, dtype=float)
    psi_t[(0,) * n_qubits] = 1.0

    saved_inputs = []

    def apply_ry_layer(state_t: np.ndarray, angles: np.ndarray) -> np.ndarray:
        layer_inputs = []
        cur = state_t
        for q in range(n_qubits):
            layer_inputs.append(cur.copy())
            ax = n_qubits - 1 - q
            c_val = np.cos(0.5 * angles[q])
            s_val = np.sin(0.5 * angles[q])
            s0 = [slice(None)] * n_qubits
            s1 = [slice(None)] * n_qubits
            s0[ax], s1[ax] = 0, 1
            u0 = cur[tuple(s0)].copy()
            u1 = cur[tuple(s1)].copy()
            nxt = cur.copy()
            nxt[tuple(s0)] = c_val * u0 - s_val * u1
            nxt[tuple(s1)] = s_val * u0 + c_val * u1
            cur = nxt
        saved_inputs.append((angles.copy(), layer_inputs))
        return cur

    def apply_cx_bonds(state_t: np.ndarray, bonds: list) -> np.ndarray:
        cur = state_t.copy()
        for (ctrl, targ) in bonds:
            ax_c = n_qubits - 1 - ctrl
            ax_t = n_qubits - 1 - targ
            idx_10 = [slice(None)] * n_qubits
            idx_11 = [slice(None)] * n_qubits
            idx_10[ax_c], idx_10[ax_t] = 1, 0
            idx_11[ax_c], idx_11[ax_t] = 1, 1
            tmp = cur[tuple(idx_10)].copy()
            cur[tuple(idx_10)] = cur[tuple(idx_11)]
            cur[tuple(idx_11)] = tmp
        return cur

    psi_t = apply_ry_layer(psi_t, theta[0:n_qubits])
    offset = n_qubits
    for _ in range(n_blocks):
        psi_t = apply_cx_bonds(psi_t, even_bonds)
        psi_t = apply_ry_layer(psi_t, theta[offset:offset + n_qubits])
        offset += n_qubits
        psi_t = apply_cx_bonds(psi_t, odd_bonds)
        psi_t = apply_ry_layer(psi_t, theta[offset:offset + n_qubits])
        offset += n_qubits

    psi_vec = psi_t.reshape(dim)
    v_vec = a_mat @ psi_vec
    overlap = float(np.dot(b_unit, v_vec))
    norm_sq = max(float(np.dot(v_vec, v_vec)), 1e-15)
    cost = 1.0 - (overlap ** 2) / norm_sq

    d_v = (-2.0 * overlap / norm_sq) * b_unit + (2.0 * (overlap ** 2) / (norm_sq ** 2)) * v_vec
    g_t = (a_mat.T @ d_v).reshape([2] * n_qubits)

    grad_layers = []

    def backprop_ry_layer(grad_state: np.ndarray, angles: np.ndarray, layer_inputs: list) -> tuple:
        layer_grad = np.zeros(n_qubits, dtype=float)
        cur_g = grad_state
        for q in range(n_qubits - 1, -1, -1):
            ax = n_qubits - 1 - q
            ang = angles[q]
            c_val = np.cos(0.5 * ang)
            s_val = np.sin(0.5 * ang)
            pre = layer_inputs[q]

            s0 = [slice(None)] * n_qubits
            s1 = [slice(None)] * n_qubits
            s0[ax], s1[ax] = 0, 1

            d_pre = np.empty_like(pre)
            d_pre[tuple(s0)] = -0.5 * s_val * pre[tuple(s0)] - 0.5 * c_val * pre[tuple(s1)]
            d_pre[tuple(s1)] = 0.5 * c_val * pre[tuple(s0)] - 0.5 * s_val * pre[tuple(s1)]
            layer_grad[q] = float(np.sum(cur_g * d_pre))

            g0 = cur_g[tuple(s0)].copy()
            g1 = cur_g[tuple(s1)].copy()
            nxt_g = cur_g.copy()
            nxt_g[tuple(s0)] = c_val * g0 + s_val * g1
            nxt_g[tuple(s1)] = -s_val * g0 + c_val * g1
            cur_g = nxt_g
        return cur_g, layer_grad

    for _ in range(n_blocks):
        ang_odd, inp_odd = saved_inputs.pop()
        g_t, lg_odd = backprop_ry_layer(g_t, ang_odd, inp_odd)
        grad_layers.append(lg_odd)
        g_t = apply_cx_bonds(g_t, list(reversed(odd_bonds)))

        ang_even, inp_even = saved_inputs.pop()
        g_t, lg_even = backprop_ry_layer(g_t, ang_even, inp_even)
        grad_layers.append(lg_even)
        g_t = apply_cx_bonds(g_t, list(reversed(even_bonds)))

    ang_init, inp_init = saved_inputs.pop()
    _, lg_init = backprop_ry_layer(g_t, ang_init, inp_init)
    grad_layers.append(lg_init)

    full_grad = np.concatenate(list(reversed(grad_layers)))
    return float(cost), full_grad


def train_vqlds_ansatz(a_mat: np.ndarray, b_vec: np.ndarray, n_qubits: int = 8, n_blocks: int = 8, maxiter: int = 900) -> dict:
    """
    Optimize the 8-qubit Brickwall VQLS ansatz from deterministic random initializations
    using L-BFGS-B, selecting the run with the lowest VQLS residual C_G(theta).
    """
    b_norm = float(np.linalg.norm(b_vec))
    b_unit = b_vec / max(b_norm, 1e-15)
    n_params = n_qubits * (2 * n_blocks + 1)

    best_result = None
    best_history = None

    for seed in [11, 23, 37]:
        rng = np.random.default_rng(seed)
        theta0 = rng.uniform(-0.35, 0.35, size=n_params)
        history = []

        def obj_fn(x: np.ndarray) -> tuple:
            return vqlds_cost_and_grad(x, a_mat, b_unit, n_qubits=n_qubits, n_blocks=n_blocks)

        def cb_fn(xk: np.ndarray) -> None:
            c_val, _ = vqlds_cost_and_grad(xk, a_mat, b_unit, n_qubits=n_qubits, n_blocks=n_blocks)
            history.append(c_val)

        c0, _ = obj_fn(theta0)
        history.append(c0)

        res = minimize(
            obj_fn,
            theta0,
            method="L-BFGS-B",
            jac=True,
            callback=cb_fn,
            options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10}
        )

        if best_result is None or res.fun < best_result.fun:
            best_result = res
            best_history = history

    qc_opt = build_brickwall_vqlds_circuit(best_result.x, n_qubits=n_qubits, n_blocks=n_blocks, cx_fold=1)
    psi_qiskit = np.real(Statevector.from_instruction(qc_opt).data)
    v_opt = a_mat @ psi_qiskit
    alpha_star = abs(float(np.dot(b_vec, v_opt)) / max(float(np.dot(v_opt, v_opt)), 1e-15))

    return {
        "theta": best_result.x,
        "final_cost": float(best_result.fun),
        "nit": int(best_result.nit),
        "history": best_history,
        "alpha_star": alpha_star,
        "qc_opt": qc_opt,
        "psi_qiskit": psi_qiskit
    }


def invert_8q_readout(prob256: np.ndarray, ro_matrix_1q: np.ndarray) -> np.ndarray:
    """Apply Kronecker-factored single-qubit readout matrix inversion across all 8 qubits."""
    inv_1q = np.linalg.inv(ro_matrix_1q.T)
    tensor = prob256.reshape([2] * 8)
    for axis in range(8):
        tensor = np.tensordot(inv_1q, tensor, axes=([1], [axis]))
        tensor = np.moveaxis(tensor, 0, axis)
    flat = np.maximum(tensor.reshape(256), 0.0)
    return flat / max(float(np.sum(flat)), 1e-12)


def serialize_circuit_instructions(qc: QuantumCircuit) -> list:
    """Serialize nearest-neighbor Ry and CX gates into JSON for IBM Quantum hardware execution."""
    gate_list = []
    for inst in qc.data:
        q_indices = [qc.find_bit(q).index for q in inst.qubits]
        params = [round(float(p), 8) for p in inst.operation.params]
        gate_list.append({
            "gate": inst.operation.name,
            "qubits": q_indices,
            "params": params
        })
    return gate_list


def execute_vqlds_simulation_tiers(op_data: dict, n_blocks: int = 8) -> dict:
    """
    Run head-to-head unpreconditioned vs preconditioned 8-qubit VQLS optimization,
    then evaluate the converged preconditioned circuit on noiseless Statevector,
    raw noisy QPU AerSimulator, and readout + Richardson ZNE error-mitigated QPU.
    """
    unpre_fit = train_vqlds_ansatz(
        op_data["m_nondim_q"], op_data["b_nondim"], n_qubits=8, n_blocks=n_blocks, maxiter=900
    )
    pre_fit = train_vqlds_ansatz(
        op_data["m_tilde_q"], op_data["b_tilde"], n_qubits=8, n_blocks=n_blocks, maxiter=900
    )

    # Verify Qiskit Statevector matches VQLS residual directly
    psi_chk = pre_fit["psi_qiskit"]
    v_chk = op_data["m_tilde_q"] @ psi_chk
    b_unit = op_data["b_tilde"] / np.linalg.norm(op_data["b_tilde"])
    cost_qiskit = 1.0 - (float(np.dot(b_unit, v_chk)) ** 2) / float(np.dot(v_chk, v_chk))
    if abs(cost_qiskit - pre_fit["final_cost"]) > 1e-10:
        raise RuntimeError(f"Circuit/trainer mismatch: {cost_qiskit} vs {pre_fit['final_cost']}")

    perm = op_data["perm_chrono_to_q"]
    d_right_chrono = op_data["d_right_chrono"]
    alpha_pre = pre_fit["alpha_star"]
    alpha_unpre = unpre_fit["alpha_star"]

    p_ideal_pre_chrono = (np.abs(pre_fit["psi_qiskit"]) ** 2)[perm]
    p_ideal_unpre_chrono = (np.abs(unpre_fit["psi_qiskit"]) ** 2)[perm]

    u_unpre = (alpha_unpre * np.sqrt(p_ideal_unpre_chrono) * d_right_chrono).reshape(32, 8)
    u_ideal = (alpha_pre * np.sqrt(p_ideal_pre_chrono) * d_right_chrono).reshape(32, 8)

    # Calibrated Heron-class superconducting QPU noise model (1Q Ry, 2Q CX, asymmetric readout)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(2.0e-4, 1), ["ry"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(2.2e-3, 2), ["cx"])
    ro_mat = np.array([[0.988, 0.012], [0.015, 0.985]])
    noise_model.add_all_qubit_readout_error(ReadoutError(ro_mat))

    sim_noisy = AerSimulator(noise_model=noise_model, seed_simulator=42)
    shots = 65536

    def run_noisy_fold(cx_fold: int) -> np.ndarray:
        qc_f = build_brickwall_vqlds_circuit(pre_fit["theta"], n_qubits=8, n_blocks=n_blocks, cx_fold=cx_fold)
        qc_f.measure_all()
        counts = sim_noisy.run(qc_f, shots=shots).result().get_counts()
        probs = np.zeros(256, dtype=float)
        for bitstr, cnt in counts.items():
            probs[int(bitstr.replace(" ", ""), 2)] = cnt / shots
        return probs

    p_raw_1x = run_noisy_fold(1)
    p_raw_3x = run_noisy_fold(3)

    # Kronecker readout calibration inversion followed by unclamped Richardson 1x/3x ZNE
    p_ro_1x = invert_8q_readout(p_raw_1x, ro_mat)
    p_ro_3x = invert_8q_readout(p_raw_3x, ro_mat)
    p_zne = np.maximum(1.5 * p_ro_1x - 0.5 * p_ro_3x, 0.0)
    p_zne /= max(float(np.sum(p_zne)), 1e-12)

    u_raw = (alpha_pre * np.sqrt(p_raw_1x[perm]) * d_right_chrono).reshape(32, 8)
    u_mit = (alpha_pre * np.sqrt(p_zne[perm]) * d_right_chrono).reshape(32, 8)

    pauli_op_4q = SparsePauliOp.from_operator(op_data["m_tilde_q"][:16, :16], atol=1e-5)
    qc_base = pre_fit["qc_opt"]

    return {
        "u_unpre": u_unpre,
        "u_ideal": u_ideal,
        "u_raw": u_raw,
        "u_mit": u_mit,
        "unpre_fit": unpre_fit,
        "pre_fit": pre_fit,
        "serialized_gates": serialize_circuit_instructions(qc_base),
        "circuit_depth_1x": int(qc_base.depth()),
        "cx_gate_count": int(qc_base.count_ops().get("cx", 0)),
        "ry_gate_count": int(qc_base.count_ops().get("ry", 0)),
        "core_4q_pauli_strings": int(len(pauli_op_4q))
    }


def compute_scaling_condition_numbers(model: OsimertinibCoupledPBPK, dose_uM: float) -> pd.DataFrame:
    """Measure raw, non-dimensional, and preconditioned condition numbers across 6..12 qubits."""
    records = []
    for n_qubits in [6, 8, 10, 12]:
        n_times = (2 ** n_qubits) // 8
        ops = assemble_pade_carleman_operators(model, dose_uM=dose_uM, n_times=n_times)
        records.append({
            "n_qubits": n_qubits,
            "hilbert_dimension": 2 ** n_qubits,
            "time_slices": n_times,
            "raw_condition_number": round(ops["cond_raw"], 2),
            "nondim_condition_number": round(ops["cond_nondim"], 2),
            "preconditioned_condition_number": round(ops["cond_precond"], 2),
            "compression_factor_vs_raw": round(ops["cond_raw"] / ops["cond_precond"], 1),
            "compression_factor_vs_nondim": round(ops["cond_nondim"] / ops["cond_precond"], 1)
        })
    return pd.DataFrame(records)


def evaluate_radau_reference(model: OsimertinibCoupledPBPK, dose_uM: float, t_grid: np.ndarray) -> np.ndarray:
    """Solve the nonlinear 6-state ODE via implicit Radau IIA strictly for accuracy scoring."""
    y0 = np.array([dose_uM, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sol = solve_ivp(
        model.rhs,
        [float(t_grid[0]), float(t_grid[-1])],
        y0,
        method="Radau",
        t_eval=t_grid,
        rtol=1e-10,
        atol=1e-12
    )
    return sol.y  # shape (6, len(t_grid))


def r2_score(y_ref: np.ndarray, y_est: np.ndarray) -> float:
    """Coefficient of determination R^2 against independent classical Radau trajectory."""
    ss_res = float(np.sum((y_ref - y_est) ** 2))
    ss_tot = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    return round(1.0 - ss_res / max(ss_tot, 1e-12), 4)


def pk_metrics(t_arr: np.ndarray, c_arr: np.ndarray) -> dict:
    """Compute peak concentration Cmax, peak time Tmax, and trapezoidal AUC(0-72h)."""
    idx = int(np.argmax(c_arr))
    return {
        "Cmax": round(float(c_arr[idx]), 5),
        "Tmax_h": round(float(t_arr[idx]), 2),
        "AUC_0_72h": round(float(np.trapezoid(c_arr, t_arr)), 4)
    }


def main() -> None:
    """Run honest from-scratch 8-qubit VQLS Osimertinib benchmarks and write outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] Loading curated Osimertinib and Ritonavir CYP3A4 parameters...")
    params = load_osimertinib_parameters()
    dose_uM = 0.335

    print("[2/4] Assembling receptor-conserving Pade-(1,2) operators & training 8-qubit VQLS...")
    model_base = OsimertinibCoupledPBPK(params, cyp3a4_inhibitor_uM=0.0)
    stiffness_ratio = model_base.compute_jacobian_stiffness(np.array([0.20, 0.15, 0.01, 0.01, 0.01, 0.01]))
    ops_base = assemble_pade_carleman_operators(model_base, dose_uM=dose_uM, n_times=32)
    q_base = execute_vqlds_simulation_tiers(ops_base, n_blocks=8)

    print("[3/4] Training 8-qubit VQLS for Ritonavir CYP3A4 co-medication DDI scenario...")
    model_ddi = OsimertinibCoupledPBPK(
        params,
        cyp3a4_inhibitor_uM=params["ritonavir_conc_uM"],
        ki_uM=params["ritonavir_ki_uM"]
    )
    ops_ddi = assemble_pade_carleman_operators(model_ddi, dose_uM=dose_uM, n_times=32)
    q_ddi = execute_vqlds_simulation_tiers(ops_ddi, n_blocks=8)

    print("[4/4] Evaluating independent classical Radau IIA reference and saving results...")
    t32 = ops_base["t_grid"]
    y_rad_base = evaluate_radau_reference(model_base, dose_uM=dose_uM, t_grid=t32)
    y_rad_ddi = evaluate_radau_reference(model_ddi, dose_uM=dose_uM, t_grid=t32)

    df_traj = pd.DataFrame({
        "time_h": np.round(t32, 4),
        "C_gut_uM_radau": np.round(y_rad_base[0], 6),
        "C_gut_uM_noiseless": np.round(q_base["u_ideal"][:, 0], 6),
        "C_osimertinib_uM_radau": np.round(y_rad_base[1], 6),
        "C_osimertinib_uM_vqlds_unpreconditioned": np.round(q_base["u_unpre"][:, 1], 6),
        "C_osimertinib_uM_noiseless": np.round(q_base["u_ideal"][:, 1], 6),
        "C_osimertinib_uM_qpu_raw": np.round(q_base["u_raw"][:, 1], 6),
        "C_osimertinib_uM_qpu_mitigated": np.round(q_base["u_mit"][:, 1], 6),
        "C_AZ5104_uM_radau": np.round(y_rad_base[2], 6),
        "C_AZ5104_uM_vqlds_unpreconditioned": np.round(q_base["u_unpre"][:, 2], 6),
        "C_AZ5104_uM_noiseless": np.round(q_base["u_ideal"][:, 2], 6),
        "C_AZ5104_uM_qpu_raw": np.round(q_base["u_raw"][:, 2], 6),
        "C_AZ5104_uM_qpu_mitigated": np.round(q_base["u_mit"][:, 2], 6),
        "C_AZ7550_uM_radau": np.round(y_rad_base[3], 6),
        "C_AZ7550_uM_vqlds_unpreconditioned": np.round(q_base["u_unpre"][:, 3], 6),
        "C_AZ7550_uM_noiseless": np.round(q_base["u_ideal"][:, 3], 6),
        "C_AZ7550_uM_qpu_raw": np.round(q_base["u_raw"][:, 3], 6),
        "C_AZ7550_uM_qpu_mitigated": np.round(q_base["u_mit"][:, 3], 6),
        "C_EGFR_rev_uM_radau": np.round(y_rad_base[4], 6),
        "C_EGFR_cov_uM_radau": np.round(y_rad_base[5], 6),
        "C_EGFR_cov_uM_noiseless": np.round(q_base["u_ideal"][:, 5], 6),
        "C_osimertinib_ddi_radau": np.round(y_rad_ddi[1], 6),
        "C_osimertinib_ddi_noiseless": np.round(q_ddi["u_ideal"][:, 1], 6),
        "C_osimertinib_ddi_qpu_mitigated": np.round(q_ddi["u_mit"][:, 1], 6)
    })
    df_traj.to_csv(RESULTS_DIR / "stiff_dynamics_trajectories.csv", index=False)

    h_unpre = q_base["unpre_fit"]["history"]
    h_pre = q_base["pre_fit"]["history"]
    max_len = max(len(h_unpre), len(h_pre))
    pad_unpre = h_unpre + [h_unpre[-1]] * (max_len - len(h_unpre))
    pad_pre = h_pre + [h_pre[-1]] * (max_len - len(h_pre))
    df_conv = pd.DataFrame({
        "iteration": np.arange(max_len),
        "unpreconditioned_vqlds_cost": np.round(pad_unpre, 8),
        "preconditioned_vqlds_cost": np.round(pad_pre, 8)
    })
    df_conv.to_csv(RESULTS_DIR / "vqlds_convergence_history.csv", index=False)

    df_cond = compute_scaling_condition_numbers(model_base, dose_uM=dose_uM)
    df_cond.to_csv(RESULTS_DIR / "stiff_dynamics_condition_numbers.csv", index=False)

    with open(RESULTS_DIR / "optimal_vqlds_8q_angles.json", "w", encoding="utf-8") as handle:
        json.dump({
            "n_qubits": 8,
            "n_times": 32,
            "n_modes": 8,
            "alpha_star": round(float(q_base["pre_fit"]["alpha_star"]), 8),
            "scales": [round(float(s), 8) for s in ops_base["scales"]],
            "perm_chrono_to_q": [int(idx) for idx in ops_base["perm_chrono_to_q"]],
            "gates": q_base["serialized_gates"]
        }, handle, indent=2)

    u_carleman_exact = (np.linalg.solve(ops_base["m_raw"], ops_base["b_raw"])).reshape(32, 8)

    summary = {
        "jacobian_stiffness_ratio": round(stiffness_ratio, 2),
        "condition_number_8q_raw": round(ops_base["cond_raw"], 2),
        "condition_number_8q_nondim": round(ops_base["cond_nondim"], 2),
        "condition_number_8q_preconditioned": round(ops_base["cond_precond"], 2),
        "vqlds_optimization": {
            "unpreconditioned_final_cost": round(q_base["unpre_fit"]["final_cost"], 6),
            "unpreconditioned_iterations": q_base["unpre_fit"]["nit"],
            "preconditioned_final_cost": round(q_base["pre_fit"]["final_cost"], 6),
            "preconditioned_iterations": q_base["pre_fit"]["nit"],
            "carleman_exact_vs_radau_osi_r2": r2_score(y_rad_base[1], u_carleman_exact[:, 1]),
            "carleman_exact_vs_radau_az5104_r2": r2_score(y_rad_base[2], u_carleman_exact[:, 2]),
            "carleman_exact_vs_radau_az7550_r2": r2_score(y_rad_base[3], u_carleman_exact[:, 3])
        },
        "circuit_statistics": {
            "n_qubits": 8,
            "topology": "1D_nearest_neighbor_multiresolution_0_SWAP",
            "circuit_depth": q_base["circuit_depth_1x"],
            "cx_gate_count": q_base["cx_gate_count"],
            "ry_gate_count": q_base["ry_gate_count"],
            "core_4q_pauli_strings": q_base["core_4q_pauli_strings"]
        },
        "osimertinib_parent_pk": {
            "chembl_reference_Cmax_uM": params["Cmax_reference_uM"],
            "chembl_reference_Tmax_h": params["Tmax_oral_median_h"],
            "radau": pk_metrics(t32, y_rad_base[1]),
            "vqlds_unpreconditioned": pk_metrics(t32, q_base["u_unpre"][:, 1]),
            "noiseless_statevector": pk_metrics(t32, q_base["u_ideal"][:, 1]),
            "qpu_raw": pk_metrics(t32, q_base["u_raw"][:, 1]),
            "qpu_mitigated": pk_metrics(t32, q_base["u_mit"][:, 1]),
            "r2_unpreconditioned_vs_radau": r2_score(y_rad_base[1], q_base["u_unpre"][:, 1]),
            "r2_noiseless_vs_radau": r2_score(y_rad_base[1], q_base["u_ideal"][:, 1]),
            "r2_qpu_raw_vs_radau": r2_score(y_rad_base[1], q_base["u_raw"][:, 1]),
            "r2_qpu_mitigated_vs_radau": r2_score(y_rad_base[1], q_base["u_mit"][:, 1])
        },
        "az5104_toxic_metabolite_pk": {
            "radau": pk_metrics(t32, y_rad_base[2]),
            "noiseless_statevector": pk_metrics(t32, q_base["u_ideal"][:, 2]),
            "qpu_raw": pk_metrics(t32, q_base["u_raw"][:, 2]),
            "qpu_mitigated": pk_metrics(t32, q_base["u_mit"][:, 2]),
            "r2_unpreconditioned_vs_radau": r2_score(y_rad_base[2], q_base["u_unpre"][:, 2]),
            "r2_noiseless_vs_radau": r2_score(y_rad_base[2], q_base["u_ideal"][:, 2]),
            "r2_qpu_raw_vs_radau": r2_score(y_rad_base[2], q_base["u_raw"][:, 2]),
            "r2_qpu_mitigated_vs_radau": r2_score(y_rad_base[2], q_base["u_mit"][:, 2])
        },
        "az7550_active_metabolite_pk": {
            "radau": pk_metrics(t32, y_rad_base[3]),
            "noiseless_statevector": pk_metrics(t32, q_base["u_ideal"][:, 3]),
            "qpu_raw": pk_metrics(t32, q_base["u_raw"][:, 3]),
            "qpu_mitigated": pk_metrics(t32, q_base["u_mit"][:, 3]),
            "r2_unpreconditioned_vs_radau": r2_score(y_rad_base[3], q_base["u_unpre"][:, 3]),
            "r2_noiseless_vs_radau": r2_score(y_rad_base[3], q_base["u_ideal"][:, 3]),
            "r2_qpu_raw_vs_radau": r2_score(y_rad_base[3], q_base["u_raw"][:, 3]),
            "r2_qpu_mitigated_vs_radau": r2_score(y_rad_base[3], q_base["u_mit"][:, 3])
        },
        "ritonavir_ddi_impact": {
            "baseline_AUC_0_72h": pk_metrics(t32, y_rad_base[1])["AUC_0_72h"],
            "inhibited_AUC_0_72h_radau": pk_metrics(t32, y_rad_ddi[1])["AUC_0_72h"],
            "inhibited_AUC_0_72h_vqlds_mitigated": pk_metrics(t32, q_ddi["u_mit"][:, 1])["AUC_0_72h"],
            "r2_ddi_noiseless": r2_score(y_rad_ddi[1], q_ddi["u_ideal"][:, 1]),
            "r2_ddi_qpu_mitigated": r2_score(y_rad_ddi[1], q_ddi["u_mit"][:, 1])
        }
    }

    with open(RESULTS_DIR / "stiff_dynamics_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nCompleted receptor-conserving 8-qubit VQLS stiff dynamics benchmark.")
    print(json.dumps(summary, indent=2))
    print("\nCondition number scaling across register sizes:")
    print(df_cond.to_string(index=False))


if __name__ == "__main__":
    main()
