"""
Readout-screened IBM Quantum hardware execution for the 8-qubit Osimertinib VQLS circuit.

This standalone script (`05_ibm_vqlds_readout_screened.py`) leaves `04_run_ibm_quantum_hardware.py`
and `results/04_ibm_hardware/` completely untouched. It reads (read-only) the prior empirical
8-qubit readout calibration from `results/04_ibm_hardware/` to exclude any transient TLS
readout-defect qubits (such as >14% flip-rate channels), screens `backend.target` across all
connected 8-qubit Heavy-Hex paths to select a low-readout-error, low-CZ-error physical chain,
and executes the 4-PUB Osimertinib VQLS batch (`ro0`, `ro1`, `fold1`, `fold3`) at higher shot
statistics (default: 16,384 shots/PUB, ~16-18 seconds QPU time on `ibm_marrakesh`).

All outputs are saved exclusively to a separate directory:
  `results/05_ibm_vqlds_screened/`
    - ibm_screened_job_manifest.json
    - ibm_screened_raw_counts.json
    - ibm_screened_vqlds_trajectories.csv
    - ibm_screened_hardware_results.json
"""

import argparse
import json
import os
from pathlib import Path

# NumPy and Pandas reconstruct physical Osimertinib & metabolite trajectories from counts
import numpy as np
import pandas as pd

# Qiskit core transpiler and Runtime V2 hardware interface (qiskit-ibm-runtime >= 0.50)
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits


PROJECT_DIR = Path(__file__).resolve().parent.parent
STIFF_RESULTS_DIR = PROJECT_DIR / "results" / "02_stiff_dynamics"
PRIOR_HW_DIR = PROJECT_DIR / "results" / "04_ibm_hardware"
RESULTS_DIR = PROJECT_DIR / "results" / "05_ibm_vqlds_screened"


def build_8q_vqlds_circuit(gate_list: list, n_qubits: int = 8) -> QuantumCircuit:
    """Rebuild the 8-qubit nearest-neighbor Brickwall VQLS circuit from serialized instructions."""
    qc = QuantumCircuit(n_qubits, name="vqlds_8q_fold1")
    for item in gate_list:
        gname = item["gate"]
        qargs = item["qubits"]
        pars = item["params"]
        if gname == "ry":
            qc.ry(pars[0], qargs[0])
        elif gname == "rx":
            qc.rx(pars[0], qargs[0])
        elif gname == "rz":
            qc.rz(pars[0], qargs[0])
        elif gname == "u":
            qc.u(pars[0], pars[1], pars[2], qargs[0])
        elif gname == "cx":
            qc.cx(qargs[0], qargs[1])
    qc.measure_all()
    return qc


def fold_isa_2q_gates(isa_qc: QuantumCircuit, fold_factor: int = 3, name: str = None) -> QuantumCircuit:
    """
    Odd-fold native two-qubit ISA gates (`cz`, `ecr`, `cx`) after transpilation,
    separating folded copies with barriers so compiler passes cannot cancel them
    while preserving the exact physical qubit chain of `isa_qc`.
    """
    folded = QuantumCircuit(*isa_qc.qregs, *isa_qc.cregs, name=name or f"{isa_qc.name}_fold{fold_factor}")
    for inst in isa_qc.data:
        if inst.operation.name in ("cz", "ecr", "cx") and fold_factor > 1:
            for k in range(fold_factor):
                if k > 0:
                    folded.barrier(*inst.qubits)
                folded.append(inst.operation, inst.qubits, inst.clbits)
        else:
            folded.append(inst.operation, inst.qubits, inst.clbits)
    return folded


def build_isa_readout_cal_circuits(isa_8q: QuantumCircuit, backend_op_names: set) -> tuple:
    """Construct all-|0> and all-|1> readout calibration ISA circuits on the exact 8 physical qubits."""
    meas_insts = [inst for inst in isa_8q.data if inst.operation.name == "measure"]
    ro0 = QuantumCircuit(*isa_8q.qregs, *isa_8q.cregs, name="vqlds_8q_ro0")
    ro1 = QuantumCircuit(*isa_8q.qregs, *isa_8q.cregs, name="vqlds_8q_ro1")

    meas_qubits = [inst.qubits[0] for inst in meas_insts]
    if "id" in backend_op_names:
        for q_phys in meas_qubits:
            ro0.id(q_phys)
    ro0.barrier(*meas_qubits)
    for inst in meas_insts:
        ro0.append(inst.operation, inst.qubits, inst.clbits)

    for q_phys in meas_qubits:
        if "x" in backend_op_names:
            ro1.x(q_phys)
        else:
            ro1.sx(q_phys)
            ro1.sx(q_phys)
    ro1.barrier(*meas_qubits)
    for inst in meas_insts:
        ro1.append(inst.operation, inst.qubits, inst.clbits)

    return ro0, ro1


def extract_pub_counts(pub_res) -> dict:
    """Extract bitstring count dictionary from a SamplerV2 PUB result."""
    if hasattr(pub_res.data, "meas"):
        return pub_res.data.meas.get_counts()
    for val in pub_res.data.values():
        if hasattr(val, "get_counts"):
            return val.get_counts()
    return pub_res.join_data().get_counts()


def detect_empirical_tls_qubits_from_04(backend_name: str, threshold: float = 0.04) -> set:
    """
    Read (read-only) `results/04_ibm_hardware/` to flag any physical qubit whose live
    readout flip rate exceeded `threshold` (4%) so closed-loop layout selection avoids it.
    """
    flagged = set()
    manifest_path = PRIOR_HW_DIR / "ibm_job_manifest.json"
    raw_path = PRIOR_HW_DIR / "ibm_raw_hardware_counts.json"
    if not (manifest_path.exists() and raw_path.exists()):
        return flagged

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        with open(raw_path, "r", encoding="utf-8") as handle:
            raw_arch = json.load(handle)

        if manifest.get("backend_name") != backend_name:
            return flagged

        phys_list = None
        for entry in manifest.get("transpiled_circuits", []):
            if entry.get("name") == "vqlds_8q_ro0" and isinstance(entry.get("physical_qubit_indices"), list):
                phys_list = entry["physical_qubit_indices"]
                break
        if not phys_list or len(phys_list) != 8:
            return flagged

        ro0 = raw_arch["counts"][0]
        ro1 = raw_arch["counts"][1]
        s0 = max(sum(ro0.values()), 1)
        s1 = max(sum(ro1.values()), 1)
        for pos in range(8):
            e01 = sum(c for b, c in ro0.items() if b.replace(" ", "")[pos] == "1") / s0
            e10 = sum(c for b, c in ro1.items() if b.replace(" ", "")[pos] == "0") / s1
            if max(e01, e10) > threshold:
                flagged.add(int(phys_list[7 - pos]))
    except Exception:
        return flagged

    return flagged


def select_readout_screened_8q_chain(backend) -> tuple:
    """
    Enumerate all connected 8-qubit simple paths on `backend.target` and select the chain
    minimizing combined 56-CZ gate infidelity + 8-qubit readout assignment error while
    excluding both calibration-flagged and empirically detected TLS readout-defect qubits.
    """
    target = backend.target
    n_phys = backend.num_qubits
    excluded = detect_empirical_tls_qubits_from_04(backend.name, threshold=0.04)

    meas_map = target["measure"] if "measure" in target.operation_names else {}
    twoq_op = "cz" if "cz" in target.operation_names else ("ecr" if "ecr" in target.operation_names else "cx")
    twoq_map = target[twoq_op] if twoq_op in target.operation_names else {}
    sx_map = target["sx"] if "sx" in target.operation_names else {}

    meas_err = np.ones(n_phys, dtype=float)
    sx_err = np.zeros(n_phys, dtype=float)
    for q in range(n_phys):
        m_prop = meas_map.get((q,))
        val_m = getattr(m_prop, "error", None) if m_prop is not None else None
        meas_err[q] = 0.50 if val_m is None else float(val_m)

        s_prop = sx_map.get((q,))
        val_s = getattr(s_prop, "error", None) if s_prop is not None else None
        sx_err[q] = 0.01 if val_s is None else float(val_s)

    adj = {q: {} for q in range(n_phys)}
    for qpair, prop in twoq_map.items():
        if len(qpair) != 2:
            continue
        u, v = int(qpair[0]), int(qpair[1])
        val_e = getattr(prop, "error", None)
        e_uv = 0.50 if val_e is None else float(val_e)
        if e_uv < 0.08 and u not in excluded and v not in excluded:
            prev = adj[u].get(v, 1.0)
            if e_uv < prev:
                adj[u][v] = e_uv
                adj[v][u] = e_uv

    best_score = 1e9
    best_path = None

    def dfs(path: list, visited: set, cur_cz_sum: float) -> None:
        nonlocal best_score, best_path
        if len(path) == 8:
            max_m = max(meas_err[q] for q in path)
            max_c = max(adj[path[k]][path[k + 1]] for k in range(7))
            penalty = 0.0
            if max_m > 0.020:
                penalty += 30.0 * (max_m - 0.020)
            if max_c > 0.012:
                penalty += 25.0 * (max_c - 0.012)
            score = (
                8.0 * cur_cz_sum
                + 14.0 * sum(meas_err[q] for q in path)
                + 25.0 * sum(sx_err[q] for q in path)
                + penalty
            )
            if score < best_score:
                best_score = score
                best_path = list(path)
            return

        last = path[-1]
        for nxt, e_uv in adj[last].items():
            if nxt not in visited and meas_err[nxt] < 0.08:
                visited.add(nxt)
                path.append(nxt)
                dfs(path, visited, cur_cz_sum + e_uv)
                path.pop()
                visited.remove(nxt)

    for start_q in range(n_phys):
        if start_q not in excluded and meas_err[start_q] < 0.04 and adj[start_q]:
            dfs([start_q], {start_q}, 0.0)

    if best_path is None:
        raise RuntimeError("Could not find a clean connected 8-qubit chain on backend.target.")

    q_meas_list = [round(float(meas_err[q]), 5) for q in best_path]
    bond_cz_list = [round(float(adj[best_path[k]][best_path[k + 1]]), 5) for k in range(7)]
    return best_path, q_meas_list, bond_cz_list, sorted(list(excluded))


def connect_ibm_service(token: str = None, channel: str = "ibm_quantum_platform", instance: str = None) -> QiskitRuntimeService:
    """Connect to QiskitRuntimeService using explicit token/CRN or saved local credentials."""
    kwargs = {"channel": channel}
    if token:
        kwargs["token"] = token
    if instance:
        kwargs["instance"] = instance
    return QiskitRuntimeService(**kwargs)


def decode_and_save_screened_results(
    raw_counts_list: list,
    vqlds_cfg: dict,
    shots: int,
    backend_name: str,
    job_id: str,
    selected_chain: list = None
) -> None:
    """
    Decode untouched 4-PUB readout-screened VQLS counts and write all outputs strictly
    inside `results/05_ibm_vqlds_screened/` without any clamping to ideal or Radau values.
    """
    counts_ro0, counts_ro1, counts_f1, counts_f3 = raw_counts_list
    s_ro0 = max(sum(counts_ro0.values()), 1)
    s_ro1 = max(sum(counts_ro1.values()), 1)
    s_f1 = max(sum(counts_f1.values()), 1)
    s_f3 = max(sum(counts_f3.values()), 1)

    eps_0_to_1 = np.zeros(8, dtype=float)
    eps_1_to_0 = np.zeros(8, dtype=float)
    for b, c in counts_ro0.items():
        bits = b.replace(" ", "")
        for pos in range(8):
            if bits[pos] == "1":
                eps_0_to_1[pos] += c / s_ro0
    for b, c in counts_ro1.items():
        bits = b.replace(" ", "")
        for pos in range(8):
            if bits[pos] == "0":
                eps_1_to_0[pos] += c / s_ro1

    inv_ro_full = np.array([[1.0]])
    for pos in range(8):
        e01 = min(max(eps_0_to_1[pos], 0.0), 0.45)
        e10 = min(max(eps_1_to_0[pos], 0.0), 0.45)
        r_mat = np.array([[1.0 - e01, e10], [e01, 1.0 - e10]], dtype=float)
        inv_ro_full = np.kron(inv_ro_full, np.linalg.inv(r_mat))

    p1 = np.zeros(256, dtype=float)
    p3 = np.zeros(256, dtype=float)
    for b, c in counts_f1.items():
        p1[int(b.replace(" ", ""), 2)] = c / s_f1
    for b, c in counts_f3.items():
        p3[int(b.replace(" ", ""), 2)] = c / s_f3

    p1_ro = np.maximum(inv_ro_full @ p1, 0.0)
    p1_ro /= max(float(np.sum(p1_ro)), 1e-12)

    p3_ro = np.maximum(inv_ro_full @ p3, 0.0)
    p3_ro /= max(float(np.sum(p3_ro)), 1e-12)

    p_zne = np.maximum(1.5 * p1_ro - 0.5 * p3_ro, 0.0)
    p_zne /= max(float(np.sum(p_zne)), 1e-12)

    t32 = np.linspace(0.0, 72.0, 32)
    alpha_star = float(vqlds_cfg.get("alpha_star", 1.0))
    scales = np.array(vqlds_cfg.get("scales", [1.0] * 8), dtype=float)
    perm = np.array(vqlds_cfg.get("perm_chrono_to_q", list(range(256))), dtype=int)
    d_right = np.tile(scales, 32)

    u_hw_raw = (alpha_star * np.sqrt(p1[perm]) * d_right).reshape(32, 8)
    u_hw_ro = (alpha_star * np.sqrt(p1_ro[perm]) * d_right).reshape(32, 8)
    u_hw_zne = (alpha_star * np.sqrt(p_zne[perm]) * d_right).reshape(32, 8)

    df_scr_traj = pd.DataFrame({
        "time_h": np.round(t32, 4),
        "C_osimertinib_uM_ibm_raw": np.round(u_hw_raw[:, 1], 6),
        "C_osimertinib_uM_ibm_ro": np.round(u_hw_ro[:, 1], 6),
        "C_osimertinib_uM_ibm_zne": np.round(u_hw_zne[:, 1], 6),
        "C_AZ5104_uM_ibm_raw": np.round(u_hw_raw[:, 2], 6),
        "C_AZ5104_uM_ibm_ro": np.round(u_hw_ro[:, 2], 6),
        "C_AZ5104_uM_ibm_zne": np.round(u_hw_zne[:, 2], 6),
        "C_AZ7550_uM_ibm_raw": np.round(u_hw_raw[:, 3], 6),
        "C_AZ7550_uM_ibm_ro": np.round(u_hw_ro[:, 3], 6),
        "C_AZ7550_uM_ibm_zne": np.round(u_hw_zne[:, 3], 6)
    })
    df_scr_traj.to_csv(RESULTS_DIR / "ibm_screened_vqlds_trajectories.csv", index=False)

    summary = {
        "backend_name": backend_name,
        "job_id": job_id,
        "shots_per_circuit": shots,
        "selected_physical_qubits": selected_chain,
        "per_bit_empirical_e01": [round(float(x), 5) for x in eps_0_to_1],
        "per_bit_empirical_e10": [round(float(x), 5) for x in eps_1_to_0],
        "mean_8q_readout_error_0_to_1": round(float(np.mean(eps_0_to_1)), 5),
        "mean_8q_readout_error_1_to_0": round(float(np.mean(eps_1_to_0)), 5),
        "osimertinib_Cmax_uM": {
            "raw": round(float(np.max(u_hw_raw[:, 1])), 5),
            "ro": round(float(np.max(u_hw_ro[:, 1])), 5),
            "zne": round(float(np.max(u_hw_zne[:, 1])), 5)
        },
        "osimertinib_AUC_0_72h": {
            "raw": round(float(np.trapezoid(u_hw_raw[:, 1], t32)), 4),
            "ro": round(float(np.trapezoid(u_hw_ro[:, 1], t32)), 4),
            "zne": round(float(np.trapezoid(u_hw_zne[:, 1], t32)), 4)
        },
        "az5104_AUC_0_72h": {
            "raw": round(float(np.trapezoid(u_hw_raw[:, 2], t32)), 4),
            "ro": round(float(np.trapezoid(u_hw_ro[:, 2], t32)), 4),
            "zne": round(float(np.trapezoid(u_hw_zne[:, 2], t32)), 4)
        },
        "az7550_AUC_0_72h": {
            "raw": round(float(np.trapezoid(u_hw_raw[:, 3], t32)), 4),
            "ro": round(float(np.trapezoid(u_hw_ro[:, 3], t32)), 4),
            "zne": round(float(np.trapezoid(u_hw_zne[:, 3], t32)), 4)
        }
    }

    radau_csv = STIFF_RESULTS_DIR / "stiff_dynamics_trajectories.csv"
    if radau_csv.exists():
        df_rad = pd.read_csv(radau_csv)
        r2_block = {}
        for label, col_rad, m_idx in [
            ("osimertinib", "C_osimertinib_uM_radau", 1),
            ("az5104", "C_AZ5104_uM_radau", 2),
            ("az7550", "C_AZ7550_uM_radau", 3),
        ]:
            y_ref = df_rad[col_rad].to_numpy(dtype=float)
            ss_tot = max(float(np.sum((y_ref - np.mean(y_ref)) ** 2)), 1e-12)
            r2_raw = 1.0 - float(np.sum((y_ref - u_hw_raw[:, m_idx]) ** 2)) / ss_tot
            r2_ro = 1.0 - float(np.sum((y_ref - u_hw_ro[:, m_idx]) ** 2)) / ss_tot
            r2_zne = 1.0 - float(np.sum((y_ref - u_hw_zne[:, m_idx]) ** 2)) / ss_tot
            r2_block[label] = {
                "raw_r2_vs_radau": round(r2_raw, 4),
                "ro_r2_vs_radau": round(r2_ro, 4),
                "zne_r2_vs_radau": round(r2_zne, 4)
            }
        summary["r2_vs_classical_radau"] = r2_block

    summary["top_basis_states_fold1"] = sorted(counts_f1.items(), key=lambda kv: kv[1], reverse=True)[:8]

    out_path = RESULTS_DIR / "ibm_screened_hardware_results.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved readout-screened IBM hardware results to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run readout-screened 8q VQLS Osimertinib circuits on IBM Quantum.")
    parser.add_argument("--token", type=str, default=os.environ.get("IBM_QUANTUM_TOKEN"), help="IBM Quantum API token")
    parser.add_argument("--channel", type=str, default="ibm_quantum_platform", choices=["ibm_quantum_platform", "ibm_cloud"])
    parser.add_argument("--instance", type=str, default=os.environ.get("IBM_QUANTUM_INSTANCE"), help="Optional CRN / instance")
    parser.add_argument("--backend", type=str, default="ibm_marrakesh", help="Target IBM backend (default: ibm_marrakesh)")
    parser.add_argument("--shots", type=int, default=16384, help="Shots per circuit PUB (default: 16384)")
    parser.add_argument("--wait", action="store_true", help="Wait synchronously for hardware queue completion")
    parser.add_argument("--fetch-job", type=str, default=None, help="Retrieve results from a previously submitted Job ID")
    parser.add_argument("--redecode-local", action="store_true", help="Re-decode from ibm_screened_raw_counts.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    vqlds_path = STIFF_RESULTS_DIR / "optimal_vqlds_8q_angles.json"
    if not vqlds_path.exists():
        raise FileNotFoundError(
            "Pre-converged VQLS parameter file not found in results/02_stiff_dynamics/. "
            "Run `python src/02_quantum_stiff_dynamics.py` first."
        )

    with open(vqlds_path, "r", encoding="utf-8") as handle:
        vqlds_cfg = json.load(handle)

    if args.redecode_local:
        raw_path = RESULTS_DIR / "ibm_screened_raw_counts.json"
        with open(raw_path, "r", encoding="utf-8") as handle:
            arch = json.load(handle)
        decode_and_save_screened_results(
            arch["counts"],
            vqlds_cfg,
            int(arch.get("shots_per_circuit", args.shots)),
            arch.get("backend_name", args.backend),
            arch.get("job_id", ""),
            arch.get("selected_physical_qubits")
        )
        return

    service = connect_ibm_service(token=args.token, channel=args.channel, instance=args.instance)

    if args.fetch_job:
        print(f"Fetching readout-screened 8q VQLS job {args.fetch_job} from IBM Quantum...")
        job = service.job(args.fetch_job)
        status = str(job.status())
        print(f"Job status: {status}")
        if "DONE" in status.upper():
            b_name = getattr(job.backend(), "name", args.backend) if job.backend() else args.backend
            man_path = RESULTS_DIR / "ibm_screened_job_manifest.json"
            sel_chain = None
            if man_path.exists():
                with open(man_path, "r", encoding="utf-8") as handle:
                    sel_chain = json.load(handle).get("selected_physical_qubits")
            pub_res = job.result()
            raw_counts_list = [extract_pub_counts(pub_res[idx]) for idx in range(4)]
            raw_archive = {
                "backend_name": b_name,
                "job_id": args.fetch_job,
                "shots_per_circuit": args.shots,
                "selected_physical_qubits": sel_chain,
                "pub_names": ["vqlds_8q_ro0", "vqlds_8q_ro1", "vqlds_8q_fold1", "vqlds_8q_fold3"],
                "counts": raw_counts_list
            }
            with open(RESULTS_DIR / "ibm_screened_raw_counts.json", "w", encoding="utf-8") as handle:
                json.dump(raw_archive, handle, indent=2)
            decode_and_save_screened_results(raw_counts_list, vqlds_cfg, args.shots, b_name, args.fetch_job, sel_chain)
        return

    backend = service.backend(args.backend) if args.backend else service.least_busy(operational=True, simulator=False, min_num_qubits=100)
    print(f"Selected IBM Quantum processor: {backend.name} ({backend.num_qubits} physical qubits)")

    best_chain, cal_meas_errs, cal_cz_errs, excluded_tls = select_readout_screened_8q_chain(backend)
    print(f"  Excluded empirical TLS readout-defect qubits from 04 run: {excluded_tls}")
    print(f"  Selected clean 8-qubit chain : {best_chain}")
    print(f"  Calibrated readout errors    : {cal_meas_errs} (mean = {np.mean(cal_meas_errs):.4f})")
    print(f"  Calibrated 2Q CZ gate errors : {cal_cz_errs} (mean = {np.mean(cal_cz_errs):.5f})")

    qc_8q = build_8q_vqlds_circuit(vqlds_cfg["gates"], vqlds_cfg["n_qubits"])
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend, initial_layout=best_chain)
    isa_8q_f1 = pm.run(qc_8q)
    isa_8q_ro0, isa_8q_ro1 = build_isa_readout_cal_circuits(isa_8q_f1, set(backend.target.operation_names))
    isa_8q_f3 = fold_isa_2q_gates(isa_8q_f1, fold_factor=3, name="vqlds_8q_fold3")

    isa_circuits = [isa_8q_ro0, isa_8q_ro1, isa_8q_f1, isa_8q_f3]
    validate_isa_circuits(isa_circuits, backend.target)

    max_s = getattr(backend, "max_shots", 100000) or 100000
    eff_shots = int(min(args.shots, max_s))

    transpile_stats = []
    for qc_isa in isa_circuits:
        ops = {str(k): int(v) for k, v in qc_isa.count_ops().items()}
        meas_phys = [int(qc_isa.find_bit(inst.qubits[0]).index) for inst in qc_isa.data if inst.operation.name == "measure"]
        transpile_stats.append({
            "name": qc_isa.name,
            "physical_qubit_indices": meas_phys,
            "isa_depth": int(qc_isa.depth()),
            "gate_counts": ops
        })
        print(f"  ISA validated {qc_isa.name:18s} | phys={meas_phys} | depth={qc_isa.depth():4d} | cz={ops.get('cz', ops.get('ecr', 0)):4d}")

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = eff_shots
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XX"
    sampler.options.dynamical_decoupling.skip_reset_qubits = True
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True

    job = sampler.run(isa_circuits, shots=eff_shots)
    job_id = job.job_id()
    print(f"\nSubmitted 4-PUB readout-screened 8q VQLS batch ({eff_shots} shots/PUB) to {backend.name}. Job ID: {job_id}")

    manifest = {
        "backend_name": backend.name,
        "backend_num_qubits": int(backend.num_qubits),
        "job_id": job_id,
        "shots": eff_shots,
        "excluded_prior_tls_qubits": excluded_tls,
        "selected_physical_qubits": best_chain,
        "calibrated_readout_errors": cal_meas_errs,
        "calibrated_2q_errors": cal_cz_errs,
        "transpiled_circuits": transpile_stats
    }
    with open(RESULTS_DIR / "ibm_screened_job_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if args.wait:
        print("Waiting for IBM Quantum hardware queue completion (Ctrl+C safe: Job ID saved in results/05_ibm_vqlds_screened/)...")
        pub_res = job.result()
        raw_counts_list = [extract_pub_counts(pub_res[idx]) for idx in range(4)]
        raw_archive = {
            "backend_name": backend.name,
            "job_id": job_id,
            "shots_per_circuit": eff_shots,
            "selected_physical_qubits": best_chain,
            "pub_names": ["vqlds_8q_ro0", "vqlds_8q_ro1", "vqlds_8q_fold1", "vqlds_8q_fold3"],
            "counts": raw_counts_list
        }
        raw_path = RESULTS_DIR / "ibm_screened_raw_counts.json"
        with open(raw_path, "w", encoding="utf-8") as handle:
            json.dump(raw_archive, handle, indent=2)
        print(f"Archived untouched screened raw hardware counts to: {raw_path}")
        decode_and_save_screened_results(raw_counts_list, vqlds_cfg, eff_shots, backend.name, job_id, best_chain)
    else:
        print(f"Job queued! Re-run with `--fetch-job {job_id} --shots {eff_shots}` once completed.")


if __name__ == "__main__":
    main()
