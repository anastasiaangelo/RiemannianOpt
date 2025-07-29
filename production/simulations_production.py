import jax
import quimb.tensor
import jax.numpy as jnp
import pandas as pd
import numpy as np
import json
import os
import yaml
import pickle
import quimb.tensor as qtn

from functools import partial
from qiskit import transpile, qpy
from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator

from mpo_lib.models.pxp_model import PXPModel1D
from mpo_lib.circuit.brickwall import BrickwallCircuit

from ropt_aqc.pxp_model import name_layered_gates, pxp_hamiltonian_sparse, generate_reference_mpo, merge_single_qubit_gates
from ropt_aqc.circuit_building import remap_layered_gate_indices, transform_layered_gates_to_qc_with_names
from ropt_aqc.comparison_methods import compress_AQC_unitary_with_checkpoint, compress_AQC_unitary_reverse_with_checkpoint, compress_HS_AQC_unitary_with_checkpoint, compress_HS_AQC_unitary_reverse_checkpointed
from ropt_aqc.brickwall_opt import plot_fidelity, optimize_swap_network_circuit_RieADAM
from ropt_aqc.brickwall_circuit import get_initial_gates
from ropt_aqc.save_model import load_reference
from ropt_aqc.tn_helpers import left_to_right_QR_sweep
from ropt_aqc.circuit_building import transform_to_bw_qc_unitarygate

from supporting_functions import NumpyEncoder

jax.config.update("jax_enable_x64", True)

output_dir = '/home/b/aag/ropt-aqc/data/raw/'


def encode_results_json(
    num_sites, num_steps, final_time,
    original_depth, initial_aqc_depth, initial_transpiled_depth, elbow_N, plateau_N, df,
    aqc_final_parameters, aqc_initial_parameters, aqc_target_mpo
):
    return {
        'params': (num_sites, num_steps, final_time),
        'original_depth': original_depth,
        'initial_aqc_depth': initial_aqc_depth,
        'initial_transpiled_depth': initial_transpiled_depth,
        'elbow_N': elbow_N,
        'plateau_N': plateau_N,
        'df': df,
        'aqc_final_parameters': aqc_final_parameters,
        'aqc_initial_parameters': aqc_initial_parameters,
        'aqc_target_mpo': aqc_target_mpo
    }


def PXP_AQC_forward_compression(num_sites, num_steps, final_time):
    method = 'AQC-forward'

    model_first_order = PXPModel1D.first_order(num_sites, final_time, num_steps)

    layered_gates = BrickwallCircuit.to_layered_gate_list(model_first_order)
    named_layered_gates = name_layered_gates(layered_gates)

    layered_gates_remapped = remap_layered_gate_indices(named_layered_gates, num_sites, qubit_direction="left-to-right")
    remapped_circuit = transform_layered_gates_to_qc_with_names(num_sites, layered_gates_remapped)
    original_depth = remapped_circuit.depth()

    transpiled_circuit = transpile(remapped_circuit, basis_gates=['cx', 'rz', 'sx'], optimization_level=3)
    initial_transpiled_depth = transpiled_circuit.depth()

    aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(remapped_circuit, parameter_name="x")
    aqc_ansatz_original = aqc_ansatz.assign_parameters(aqc_initial_parameters)
    initial_aqc_depth = aqc_ansatz.depth()

    # Save full ansatz
    qpy_path = os.path.join(output_dir, f"aqc_ansatz_{num_sites}q_{final_time}t_{num_steps}steps_forward.qpy")
    with open(qpy_path, "wb") as f:
        qpy.dump(aqc_ansatz, f)

    hamiltonian = pxp_hamiltonian_sparse(num_sites)
    aqc_target_log2_num_trotter_steps = 8
    aqc_target_mpo, _, target_circ = generate_reference_mpo(final_time, aqc_target_log2_num_trotter_steps, hamiltonian)

    simulator_settings = QuimbSimulator(
        partial(quimb.tensor.Circuit, gate_contract="split-gate"),
        autodiff_backend="jax"
    )

    # Compression
    N_vals = range(1, initial_aqc_depth + 1)
    checkpoint_path = os.path.join(output_dir, f"checkpoint_PXP_AQC_forward_{num_sites}q_{final_time}t_{num_steps}.pkl")
    df, aqc_truncated, aqc_params_truncated, aqc_final_parameters = compress_AQC_unitary_with_checkpoint(
        aqc_ansatz,
        aqc_initial_parameters,
        aqc_target_mpo,
        N_vals,
        simulator_settings,
        num_steps,
        checkpoint_path=checkpoint_path
    )

    # Save CSV
    file_path = os.path.join(output_dir, f"AQC_unitary_PXP_results_{num_sites}q_{final_time}_{num_steps}steps_forward.csv")
    df.to_csv(file_path, index=False)

    # Plot
    elbow_N, plateau_N = plot_fidelity(df, 'Final_fidelity', final_time, 1e-3, method, 'PXP', save_path=True)

    # JSON metadata
    result = encode_results_json(
        num_sites, num_steps, final_time,
        original_depth, initial_aqc_depth, initial_transpiled_depth,
        elbow_N, plateau_N, df,
        aqc_final_parameters, aqc_initial_parameters, aqc_target_mpo
    )

    json_path = os.path.join(output_dir, f"{method}-{num_sites}sites-{num_steps}steps-{final_time}t-forward.json")
    with open(json_path, "w") as f:
        json.dump(result, f, cls=NumpyEncoder)


def PXP_AQC_reverse_compression(num_sites, num_steps, final_time):
    method = 'AQC-reverse'

    model_first_order = PXPModel1D.first_order(num_sites=num_sites, final_time=final_time, num_steps=num_steps)

    layered_gates = BrickwallCircuit.to_layered_gate_list(model_first_order)
    named_layered_gates = name_layered_gates(layered_gates)
    layered_gates_remapped = remap_layered_gate_indices(named_layered_gates, num_sites, qubit_direction="left-to-right")
    remapped_circuit = transform_layered_gates_to_qc_with_names(num_sites, layered_gates_remapped)

    original_depth = remapped_circuit.depth()
    transpiled_circuit = transpile(remapped_circuit, basis_gates=['cx', 'rz', 'sx'], optimization_level=3)
    initial_transpiled_depth = transpiled_circuit.depth()

    aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(
        remapped_circuit, parameter_name="x"
    )
    aqc_ansatz_original = aqc_ansatz.assign_parameters(aqc_initial_parameters)
    initial_aqc_depth = aqc_ansatz.depth()

    hamiltonian = pxp_hamiltonian_sparse(num_sites)
    aqc_target_log2_num_trotter_steps = 8
    aqc_target_mpo, aqc_target_step_circuit, target_circ = generate_reference_mpo(
        final_time, aqc_target_log2_num_trotter_steps, hamiltonian
    )

    # set up quimb
    simulator_settings = QuimbSimulator(
        quimb_circuit_factory=partial(qtn.CircuitMPS, gate_contract="split-gate"),
        autodiff_backend="explicit",
        progbar=False,
    )

    checkpoint_path = os.path.join(output_dir, f"checkpoint_PXP_AQC_reverse_{num_sites}q_{final_time}t_{num_steps}.pkl")
    df, aqc_truncated, aqc_params_truncated, aqc_final_parameters = compress_AQC_unitary_reverse_with_checkpoint(
        aqc_ansatz,
        aqc_initial_parameters,
        initial_aqc_depth,
        target_circ,
        simulator_settings,
        fidelity_threshold=0.97,
        checkpoint_path=checkpoint_path,
    )

    file_path_PXP = os.path.join(output_dir, f'AQC_unitary_PXP_results_{num_sites}q_{final_time}_{num_steps}steps_reverse.csv')
    df.to_csv(file_path_PXP, index=False, header=True)

    qpy_path = os.path.join(output_dir, f"aqc_ansatz_{num_sites}q_{final_time}t_{num_steps}steps_reverse.qpy")
    with open(qpy_path, "wb") as f:
        qpy.dump(aqc_truncated, f)

    df_sorted = df.sort_values("circuit_layers", ascending=True)
    elbow_N, plateau_N = plot_fidelity(df_sorted, 'Final_fidelity', final_time, 1e-3, method, 'PXP', save_path=True)

    result = encode_results_json(
        num_sites, num_steps, final_time,
        original_depth, initial_aqc_depth, initial_transpiled_depth,
        elbow_N, plateau_N, df,
        aqc_final_parameters, aqc_initial_parameters, aqc_target_mpo
    )

    with open(os.path.join(output_dir, f"{method}-{num_sites}sites-{num_steps}steps-{final_time}t-reverse.json"), 'w') as f:
        json.dump(result, f, cls=NumpyEncoder)


def PXP_hybrid_forward_compression(num_sites, num_steps, final_time):
    num_sites = int(num_sites)
    num_steps = int(num_steps)
    final_time = float(final_time)

    method = 'Hybrid-forward'
    system = 'pxp'

    model_first_order = PXPModel1D.first_order(num_sites=num_sites, num_steps=num_steps, final_time=final_time)

    layered_gates = BrickwallCircuit.to_layered_gate_list(model_first_order)
    layered_gates_remapped = remap_layered_gate_indices(layered_gates, num_sites)
    remapped_circuit = transform_layered_gates_to_qc_with_names(num_sites, layered_gates_remapped)
    merged_layers = merge_single_qubit_gates(layered_gates_remapped, with_names=True, use_TN=False)
    remapped_circuit_merged = transform_layered_gates_to_qc_with_names(num_sites, merged_layers)

    transpiled_circuit = transpile(remapped_circuit, basis_gates=['cx', 'rz', 'sx'], optimization_level=3)
    initial_transpiled_depth = transpiled_circuit.depth()
    original_depth = remapped_circuit.depth()

    aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(remapped_circuit_merged, parameter_name="x")
    aqc_ansatz_original = aqc_ansatz.assign_parameters(aqc_initial_parameters)
    initial_aqc_depth = aqc_ansatz_original.depth()

    gates = get_initial_gates(n_sites=num_sites, t=final_time, n_repetitions=num_steps, degree=1, hamiltonian=system, n_id_layers=0, use_TN=True)

    repo_root = "/home/b/aag/ropt-aqc/"
    config_file = os.path.join(repo_root, "run", system, "configs", "config.yml")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    config['reference_dir'] = f'/home/b/aag/ropt-aqc/run/{system}/reference'

    U_ref, t, *_ = load_reference(config['reference_dir'], config['n_sites'], config['n_repetitions'], config['t'])
    U_ref = left_to_right_QR_sweep(U_ref, get_norm=False, normalize=config['normalize_reference'])

    gates_optimised, err_iter = optimize_swap_network_circuit_RieADAM(config, U_ref, gates)
    gates_optimised = gates_optimised.reshape((len(gates_optimised), 4, 4))

    with open(os.path.join(output_dir, f"optimized_gates{num_sites}q_{num_steps}steps_{final_time}t_forward.pkl"), "wb") as f:
        pickle.dump((gates_optimised, err_iter), f)

    qc_optimised = transform_to_bw_qc_unitarygate(num_sites, num_steps, system=system, gates=gates_optimised)

    simulator_settings = QuimbSimulator(partial(qtn.CircuitMPS, gate_contract="split-gate"), autodiff_backend="jax")

    pxp_hamiltonian = pxp_hamiltonian_sparse(num_sites)
    aqc_target_log2_num_trotter_steps = 8
    aqc_target_mpo, aqc_target_step_circuit, target_circ = generate_reference_mpo(final_time, aqc_target_log2_num_trotter_steps, pxp_hamiltonian)

    aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(qc_optimised, parameter_name="x")
    aqc_ansatz_original = aqc_ansatz.assign_parameters(aqc_initial_parameters)
    aqc_depth = int(aqc_ansatz_original.depth())

    # === CHECKPOINT HANDLING ===
    checkpoint_path = os.path.join(output_dir, f"checkpoint_forward_{num_sites}q_{num_steps}steps_{final_time}t.pkl")
    file_path_PXP_hybrid = os.path.join(output_dir, f'Hybrid_unitary_PXP_results_{num_sites}q_{final_time}_{num_steps}steps_forward.csv')

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            df, aqc_ansatz, aqc_initial_parameters, aqc_final_parameters = pickle.load(f)
    else:
        N_vals = range(1, aqc_depth + 1)
        df, aqc_ansatz, aqc_initial_parameters, aqc_final_parameters = compress_HS_AQC_unitary_with_checkpoint(
            gates_optimised, config, simulator_settings, aqc_target_mpo, N_vals, checkpoint_path=checkpoint_path
        )

    df.to_csv(file_path_PXP_hybrid, index=False, header=True)

    elbow_N, plateau_N = plot_fidelity(df, 'Final_fidelity', final_time, 1e-3, method, system, save_path=True)

    result = encode_results_json(
        num_sites, num_steps, final_time, original_depth, initial_aqc_depth,
        initial_transpiled_depth, elbow_N, plateau_N,
        df, aqc_final_parameters, aqc_initial_parameters, aqc_target_mpo
    )

    with open(os.path.join(output_dir, f"{method}-{num_sites}sites-{num_steps}steps-{final_time}t-forward.json"), 'w') as f:
        json.dump(result, f, cls=NumpyEncoder)


def PXP_hybrid_reverse_compression(num_sites, num_steps, final_time):
    num_sites = int(num_sites)
    num_steps = int(num_steps)
    final_time = float(final_time)

    method = 'Hybrid-reverse'
    system = 'pxp'

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    gates_path = os.path.join(checkpoint_dir, f"gates_{num_sites}q_{num_steps}steps_{final_time}t.pkl")
    df_path = os.path.join(checkpoint_dir, f"fidelity_df_{num_sites}q_{num_steps}steps_{final_time}t.csv")
    qpy_path = os.path.join(output_dir, f"aqc_ansatz_hybrid_{num_sites}q_{final_time}t_{num_steps}steps_reverse.qpy")
    json_path = os.path.join(output_dir, f"{method}-{num_sites}sites-{num_steps}steps-{final_time}t-reverse.json")

    # === Load model and remap ===
    model_first_order = PXPModel1D.first_order(num_sites, final_time, num_steps)
    layered_gates = BrickwallCircuit.to_layered_gate_list(model_first_order)
    layered_gates_remapped = remap_layered_gate_indices(layered_gates, num_sites)
    remapped_circuit = transform_layered_gates_to_qc_with_names(num_sites, layered_gates_remapped)
    merged_layers = merge_single_qubit_gates(layered_gates_remapped, with_names=True, use_TN=False)
    remapped_circuit_merged = transform_layered_gates_to_qc_with_names(num_sites, merged_layers)
    transpiled_circuit = transpile(remapped_circuit, basis_gates=['cx', 'rz', 'sx'], optimization_level=3)
    initial_transpiled_depth = transpiled_circuit.depth()
    original_depth = remapped_circuit.depth()

    aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(remapped_circuit_merged, parameter_name="x")
    initial_aqc_depth = aqc_ansatz.assign_parameters(aqc_initial_parameters).depth()

    # === Load config ===
    repo_root = "/home/b/aag/ropt-aqc/"
    config_path = os.path.join(repo_root, "run", system, "configs", "config.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config['reference_dir'] = f"/home/b/aag/ropt-aqc/run/{system}/reference"

    # === Load or compute optimised gates ===
    if os.path.exists(gates_path):
        with open(gates_path, "rb") as f:
            gates_optimised, err_iter = pickle.load(f)
    else:
        gates = get_initial_gates(num_sites, final_time, num_steps, degree=1, hamiltonian=system, n_id_layers=0, use_TN=True)
        U_ref, *_ = load_reference(config['reference_dir'], config['n_sites'], config['n_repetitions'], config['t'])
        U_ref = left_to_right_QR_sweep(U_ref, get_norm=False, normalize=config['normalize_reference'])

        gates_optimised, err_iter = optimize_swap_network_circuit_RieADAM(config, U_ref, gates)
        gates_optimised = gates_optimised.reshape((len(gates_optimised), 4, 4))
        with open(gates_path, "wb") as f:
            pickle.dump((gates_optimised, err_iter), f)

    # === Transform to quantum circuit ===
    qc_optimised = transform_to_bw_qc_unitarygate(num_sites, num_steps, system=system, gates=gates_optimised)

    # === Compress or resume ===
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        with open(qpy_path, "rb") as f:
            aqc_truncated = qpy.load(f)
        # TODO: optionally load final parameters too
        elbow_N, plateau_N = plot_fidelity(df, 'Final_fidelity', final_time, 1e-3, method, system, save_path=True)
    else:
        simulator_settings = QuimbSimulator(
            quimb_circuit_factory=partial(qtn.CircuitMPS, gate_contract="split-gate"),
            autodiff_backend="explicit",
            progbar=False,
        )
        pxp_hamiltonian = pxp_hamiltonian_sparse(num_sites)
        aqc_target_log2_num_trotter_steps = 8
        aqc_target_mpo, aqc_target_step_circuit, target_circ = generate_reference_mpo(final_time, aqc_target_log2_num_trotter_steps, pxp_hamiltonian)

        aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(qc_optimised, parameter_name="x")
        aqc_ansatz_original = aqc_ansatz.assign_parameters(aqc_initial_parameters)
        aqc_depth = aqc_ansatz_original.depth()

        df, aqc_truncated, aqc_final_parameters_reverse = compress_HS_AQC_unitary_reverse_checkpointed(
            gates_optimised, config, simulator_settings, aqc_depth, target_circ, fidelity_threshold=0.98
        )

        df.to_csv(df_path, index=False)
        with open(qpy_path, "wb") as f:
            qpy.dump(aqc_truncated, f)

        elbow_N, plateau_N = plot_fidelity(df, 'Final_fidelity', final_time, 1e-3, method, system, save_path=True)

    # === Save final result ===
    result = encode_results_json(
        num_sites, num_steps, final_time, original_depth, initial_aqc_depth,
        initial_transpiled_depth, elbow_N, plateau_N, df,
        aqc_final_parameters_reverse, aqc_initial_parameters, aqc_target_mpo
    )

    with open(json_path, 'w') as f:
        json.dump(result, f, cls=NumpyEncoder)
