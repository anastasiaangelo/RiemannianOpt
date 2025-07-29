import datetime
import copy
import os
import pickle
import jax.numpy as jnp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from copy import deepcopy
from scipy.optimize import OptimizeResult, minimize

from qiskit import transpile
from qiskit.quantum_info import Statevector, state_fidelity 

from qiskit_addon_aqc_tensor.objective import OneMinusFidelity, MaximizeUnitaryFidelity
from qiskit_addon_aqc_tensor.simulation import compute_overlap, tensornetwork_from_circuit
from qiskit_addon_aqc_tensor.ansatz_generation import generate_ansatz_from_circuit

from ropt_aqc.circuit_building import truncate_ansatz, truncate_parameters, truncate_qiskit_circuit, transform_to_bw_qc_unitarygate, decompose_unitary_gate
from ropt_aqc.brickwall_circuit import get_gates_per_layer
from ropt_aqc.brickwall_opt import optimize_swap_network_circuit_RieADAM
from ropt_aqc.tn_brickwall_methods import contract_layers_of_swap_network_with_mpo, get_id_mpo
from ropt_aqc.tn_helpers import convert_mpo_to_mps, get_left_canonical_mps, inner_product_mps


def compute_error_mpo(mpo1, mpo2):
    '''
    Compute the Frobenius norm and Hilbert-Schmidt test.
    '''
    # Convert MPOs to MPSs
    mps1, mps2 = convert_mpo_to_mps(mpo1), convert_mpo_to_mps(mpo2)
    # Normalize the MPS
    mps1_nrmd = get_left_canonical_mps(mps1, normalize=True, get_norm=False)
    mps2_nrmd = get_left_canonical_mps(mps2, normalize=True, get_norm=False)
    # Compute overlap
    tr = inner_product_mps(mps1_nrmd, mps2_nrmd)
    err = 2 - 2*tr.real  # Frobenius norm
    return err


def compress_HS(gates, config, U_ref, N_vals, qc_initial=None):
    """
    Truncate an quanutm circuit with brickwall layering for N layers and evaluate fidelity for compression.

    Args:
        gates (list): A list of gates representing the quantum circuit.
        config (dict): A dictionary containing the configuration parameters for the compression.
        U_ref (array): The reference unitary matrix to which the compressed circuit should be optimized.
        qc_initial (QuantumCircuit): The initial quantum circuit to be compressed.
        N_vals (list): A list of integers representing the number of layers to be used in the compression.

    Returns:
        pandas.DataFrame: A DataFrame containing the compression results for each value of N in N_vals.
    """

    data = []

    for N in N_vals:
        print(f"\n=== Optimizing with N = {N} layers ===")

        # 1. Truncate the brickwall ansatz to N layers
        gates_per_layer, _, layer_is_odd = get_gates_per_layer(
            gates, n_sites=config['n_sites'], degree=config['degree'], n_repetitions=config['n_repetitions'], n_layers=N, hamiltonian=config['hamiltonian']
        )
        gates_layer_structure = [len(layer) for layer in gates_per_layer[:N]]

        gates_subset = copy.deepcopy(gates_per_layer[:N])
        layer_is_odd_subset = copy.deepcopy(layer_is_odd[:N]) 
        layer_is_odd_subset = layer_is_odd[:N]
        gates_flat = jnp.asarray(list(chain(*copy.deepcopy(gates_subset))))

        # 2. Optimize only these layers against the full U_ref
        gates_optimised, err_iter = optimize_swap_network_circuit_RieADAM(config, U_ref, gates_flat)

        gates_optimised_reshaped = gates_optimised.reshape((len(gates_optimised), 4, 4))
        err_init = err_iter[0]
        err_opt = jnp.min(jnp.asarray(err_iter))

        # 3. Reconstruct per-layer format Use original 2-qubit gate shape
        gates_per_layer_optimised = []
        ptr = 0
        for layer in gates_subset:
            n_gates = len(layer)
            gates_per_layer_optimised.append(gates_optimised[ptr:ptr + n_gates])
            ptr += n_gates

        # 4. Build the optimised circuit
        qc_optimised = transform_to_bw_qc_unitarygate(
            num_sites=config['n_sites'], n_repetitions=config['n_repetitions'], system=config['hamiltonian'], gates=gates_optimised_reshaped
        )

        # 5. Evaluate statevector fidelity (optional, can be memory intensive)
        if qc_initial:
            try:
                state_initial = Statevector.from_instruction(qc_initial)
                state_optimized = Statevector.from_instruction(qc_optimised)
                state_fid = state_fidelity(state_initial, state_optimized)
                print(f"Statevector fidelity: {state_fid}")
            except Exception as e:
                print(f"Statevector fidelity skipped for N={N} due to: {e}")
                state_fid = None
        else:
            state_fid = None
            
        # 6. Evaluate MPO fidelity and error
        mpo_id = get_id_mpo(config['n_sites'])
        mpo_out = contract_layers_of_swap_network_with_mpo(
            mpo_id, gates_per_layer_optimised, layer_is_odd_subset,
            layer_is_left=True, max_bondim=128, get_norm=False
        )

        # After building mpo_out:
        mps_opt = convert_mpo_to_mps(mpo_out)
        _, norm_opt = get_left_canonical_mps(mps_opt, normalize=False, get_norm=True)

        # Normalize original MPO tensors by sqrt(norm_opt)
        mpo_out = [T / jnp.sqrt(norm_opt) for T in mpo_out]

        
        # mps_W = convert_mpo_to_mps(mpo_out)
        # mps_U = convert_mpo_to_mps(U_ref)

        # # Canonicalize both (no normalization yet)
        # _, norm_W = get_left_canonical_mps(mps_W, normalize=False, get_norm=True)
        # _, norm_U = get_left_canonical_mps(mps_U, normalize=False, get_norm=True)

        # # Normalize tensors
        # mps_W_normalized = [t / jnp.sqrt(norm_W) for t in mps_W]
        # mps_U_normalized = [t / jnp.sqrt(norm_U) for t in mps_U]

        # # Compute overlap
        # overlap = inner_product_mps(mps_W_normalized, mps_U_normalized)
        
        error_mpo = compute_error_mpo(mpo_out, U_ref)

        # MPO fidelity
        mps_ref = convert_mpo_to_mps(U_ref)
        mps_optimised = convert_mpo_to_mps(mpo_out)
        # Normalize reference
        mps_ref, norm_ref = get_left_canonical_mps(mps_ref, normalize=False, get_norm=True)
        mps_ref_normalized = [t / norm_ref for t in mps_ref]
        
        mps_opt_normalized = get_left_canonical_mps(mps_optimised, normalize=True, get_norm=False)

        # Compute overlap and fidelity
        overlap = inner_product_mps(mps_ref_normalized, mps_opt_normalized)
        fidelity_mpo = overlap.real**2

        numerator = inner_product_mps(mps_ref, mps_optimised).real
        norm1 = inner_product_mps(mps_ref, mps_ref).real
        norm2 = inner_product_mps(mps_optimised, mps_optimised).real

        print("Inner product:", numerator)
        print("Norms squared:", norm1, norm2)
        print("Fidelity (by hand):", (numerator ** 2) / (norm1 * norm2))

        # try:
        #     fid_vs_initial = state_fidelity(psi0, state_optimized)
        # except Exception:
        #     fid_vs_initial = None

        print(f"Frobenius error: {error_mpo}")
        print(f"MPO fidelity: {fidelity_mpo}")
        # print(f"Fidelity vs initial state: {fid_vs_initial}")

        # 7. Store results
        data.append({
            "Trotter_steps": config['n_repetitions'],
            "circuit_layers": N,
            "statevector_fidelity": state_fid,
            "frobenius_error": error_mpo,
            "mpo_fidelity": fidelity_mpo,
            "err_init/err_opt": err_init/err_opt
            # "fidelity_vs_initial": fid_vs_initial
        })

    # Optional: print as table
    print("\n\n=== Compression Results Table ===")
    print("Layers | Statevector Fidelity | Frobenius Error | MPO Fidelity")
    for r in data:
        print(f"{r['circuit_layers']:>6} | {r['statevector_fidelity']!s:>21} | {r['frobenius_error']:.6f} | {r['mpo_fidelity']:.6f}")
    
    df = pd.DataFrame(data)

    return df, gates_optimised_reshaped


def compress_AQC(aqc_ansatz, aqc_good_circuit, aqc_comparison_circuit, aqc_target_circuit, aqc_initial_parameters, N_vals, simulator_settings, aqc_target_mps, n_repetitions):
    """
        Truncate an AQC circuit for N layers and evlualte fidelity for comrpession.

        Parameters:
        aqc_ansatz (qiskit.QuantumCircuit): The ansatz circuit to be compressed.
        aqc_good_circuit (qiskit.QuantumCircuit): The good circuit to be used as a reference.
        aqc_comparison_circuit (qiskit.QuantumCircuit): The comparison circuit to be used for evaluation.
        aqc_target_circuit (qiskit.QuantumCircuit): The target circuit to be optimized towards.
        aqc_initial_parameters (list): The initial parameters for the ansatz circuit.
        N_vals (list): The number of layers to be used for compression.
        simulator_settings (dict): The settings for the simulator.
        aqc_target_mps (tensornetwork.MPS): The target MPS to be optimized towards.
        n_repetitions (int): The number of Trotter steps to be used for optimization.

        Returns:
        pandas.DataFrame: A dataframe containing the compression results.
    """
    data = []

    last_aqc_truncated = None
    last_aqc_final_parameters = None

    for N in N_vals:
        print(f"\n=== Optimizing with N = {N} layers ===")

        # Truncate circuit, ansatz and initial parameters
        aqc_truncated = truncate_ansatz(aqc_ansatz, N)
        aqc_truncated.draw('mpl')

        aqc_truncated, aqc_params_truncated = truncate_parameters(aqc_ansatz, aqc_initial_parameters, N)
        print(f"Parameters in truncated ansatz: {aqc_truncated.parameters}")
        print(f"Truncated parameter values: {aqc_params_truncated}")
        aqc_good_truncated = truncate_qiskit_circuit(aqc_good_circuit, N)
        aqc_good_truncated.draw("mpl")

        print(f"AQC Comparison circuit: depth {aqc_comparison_circuit.depth()}")
        print(f"Target circuit:         depth {aqc_target_circuit.depth()}")
        print(f"Ansatz circuit:         depth {aqc_ansatz.depth()}, with {len(aqc_initial_parameters)} parameters")
        print(f"Ansatz circuit sliced:  depth {aqc_truncated.depth()}, with {len(aqc_params_truncated)} parameters")
        
        # evaluate the fidelity between the state prepared by the initial cirucit and the target state
        good_mps = tensornetwork_from_circuit(aqc_good_truncated, simulator_settings)
        starting_fidelity = abs(compute_overlap(good_mps, aqc_target_mps)) ** 2
        print("Starting fidelity:", starting_fidelity)

        if len(aqc_params_truncated) == 0:
            print("⚠️ Skipping optimization: no parameters used at this depth.")
            final_fid = starting_fidelity
            data.append({
                "Trotter_steps": n_repetitions,
                "circuit_layers": N,
                "Starting_fidelity": starting_fidelity,
                "Final_fidelity": final_fid
            })
            continue 

        # Setting values for the optimization
        aqc_stopping_fidelity = 1
        aqc_max_iterations = 500

        stopping_point = 1.0 - aqc_stopping_fidelity
        objective = OneMinusFidelity(aqc_target_mps, aqc_truncated, simulator_settings)

        def callback(intermediate_result: OptimizeResult):
            fidelity = 1 - intermediate_result.fun
            print(f"{datetime.datetime.now()} Intermediate result: Fidelity {fidelity:.8f}")
            if intermediate_result.fun < stopping_point:
                # Good enough for now
                raise StopIteration
        try:
            result = minimize(
                objective,
                aqc_params_truncated,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": aqc_max_iterations},
                callback=callback,
            )
            if result.status not in (0, 1, 99):
                raise RuntimeError(f"Optimization failed: {result.message} (status={result.status})")

            aqc_final_parameters = result.x
            final_fidelity = 1 - result.fun

        except Exception as e:
            print(f"⚠️ Optimization failed for N={N}: {e}")
            final_fidelity = starting_fidelity  # fallback

        # Store results
        data.append({
            "Trotter_steps": n_repetitions,
            "circuit_layers": N,
            "Starting_fidelity": starting_fidelity,
            "Final_fidelity": 1 - result.fun
        })

    # Optional: print as table
    print("\n\n=== Compression Results Table ===")
    print("Layers | Starting Fidelity | Final Fidelity")
    for r in data:
        print(f"{r['circuit_layers']:>6} | {r['Starting_fidelity']!s:>21} | {r['Final_fidelity']:.6f}")
   
    df = pd.DataFrame(data)
    
    return df, aqc_truncated, aqc_initial_parameters, aqc_final_parameters


def compress_HS_AQC(gates_optimised, config, simulator_settings, aqc_target_mps, aqc_comparison_circuit, aqc_target_circuit, N_vals, terms):
    """
        Truncate an AQC circuit, warm started with Hilbert Schmidt optimisation for error, for N layers and evlualte fidelity for comrpession.

        Parameters:
        gates_optimised (list): A list of gates optimised using Hilbert Schmidt optimisation for error.
        config (dict): A dictionary containing the configuration settings for the compression.
        simulator_settings (dict): The settings for the simulator.
        aqc_target_mps (tensornetwork.MPS): The target MPS to be optimized towards.
        aqc_comparison_circuit (qiskit.QuantumCircuit): The comparison circuit to be used for evaluation.
        aqc_target_circuit (qiskit.QuantumCircuit): The target circuit to be optimized towards.
        N_vals (list): The number of layers to be used for compression.

        Returns:
        pandas.DataFrame: A dataframe containing the compression results.
        AQC ansatz
        AQC initial parameters
    """
    data = []
    gates_per_layer, _, _ = get_gates_per_layer(
        gates_optimised, n_sites=config['n_sites'], degree=config['degree'], n_repetitions=config['n_repetitions'], hamiltonian=config['hamiltonian']
    )

    flat_gates = list(chain(*gates_per_layer))

    bw_qc = transform_to_bw_qc_unitarygate(
        num_sites=config['n_sites'],
        system=config['hamiltonian'],
        gates=flat_gates,
        pauli_terms=terms
    )

    data = []
    gates_per_layer, _, _ = get_gates_per_layer(
        gates_optimised, n_sites=config['n_sites'], degree=config['degree'], n_repetitions=config['n_repetitions'], hamiltonian=config['hamiltonian']
    )

    flat_gates = list(chain(*gates_per_layer))

    bw_qc = transform_to_bw_qc_unitarygate(
        num_sites=config['n_sites'],
        system=config['hamiltonian'],
        gates=flat_gates,
        pauli_terms=terms
    )
    
    # Now decompose before generating the ansatz
    decomposed_circuit = decompose_unitary_gate(bw_qc)  # <-- this yields native gates
    aqc_ansatz_full, aqc_initial_parameters_full = generate_ansatz_from_circuit(decomposed_circuit)
    
    decomposed_circuit = transpile(decomposed_circuit, basis_gates=['rx', 'ry', 'rz', 'cx', 'id'])
    good_mps = tensornetwork_from_circuit(decomposed_circuit, simulator_settings)
    starting_fidelity = abs(compute_overlap(good_mps, aqc_target_mps)) ** 2
    print("Starting fidelity:", starting_fidelity)

    print(f"AQC Comparison circuit: depth {aqc_comparison_circuit.depth()}")
    print(f"Target circuit:         depth {aqc_target_circuit.depth()}")
    print(f"Hybrid ansatz circuit:  depth {aqc_ansatz_full.depth()}, with {len(aqc_initial_parameters_full)} parameters")
    

    for N in N_vals:
        print(f"\n=== Optimizing with N = {N} layers ===")

        # Truncate ansatz to N layers
        aqc_truncated = truncate_ansatz(aqc_ansatz_full, N)
        aqc_truncated, aqc_params_truncated = truncate_parameters(
            aqc_ansatz_full, aqc_initial_parameters_full, N
        )
        aqc_params_truncated = np.random.permutation(aqc_params_truncated)  # shuffle to avoid structure

        
        # Setting values for the optimization
        aqc_stopping_fidelity = 1
        aqc_max_iterations = 500

        stopping_point = 1.0 - aqc_stopping_fidelity
        objective = OneMinusFidelity(aqc_target_mps, aqc_truncated, simulator_settings)

        def callback(intermediate_result: OptimizeResult):
            fidelity = 1 - intermediate_result.fun
            print(f"{datetime.datetime.now()} Intermediate result: Fidelity {fidelity:.8f}")
            if intermediate_result.fun < stopping_point:
                # Good enough for now
                raise StopIteration
            
        result = minimize(
            objective,
            aqc_params_truncated,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": aqc_max_iterations},
            callback=callback,
        )
        if result.fun == 1.0 or result.status not in (0, 1, 99):
            print(f"❌ Optimization failed for N={N}, retrying with random init")
            new_params = np.random.uniform(-np.pi, np.pi, len(aqc_params_truncated))
            result = minimize(objective, new_params, ...)

        if result.status not in (
            0,
            1,
            99,
        ):  # 0 => success; 1 => max iterations reached; 99 => early termination via StopIteration
            raise RuntimeError(
                f"Optimization failed: {result.message} (status={result.status})"
            )

        print(f"Done after {result.nit} iterations.")
        aqc_final_parameters = result.x

        # Store results
        data.append({
            "Trotter_steps": config['n_repetitions'],
            "circuit_layers": N,
            "Starting_fidelity": starting_fidelity,
            "Final_fidelity": 1 - result.fun
        })

    # Optional: print as table
    print("\n\n=== Compression Results Table ===")
    print("Layers | Starting Fidelity | Final Fidelity")
    for r in data:
        print(f"{r['circuit_layers']:>6} | {r['Starting_fidelity']!s:>21} | {r['Final_fidelity']:.6f}")

    df = pd.DataFrame(data)
    return df, aqc_truncated, aqc_params_truncated, aqc_final_parameters


def compress_AQC_unitary(aqc_ansatz, aqc_initial_parameters, aqc_target_mpo, N_vals, simulator_settings, n_repetitions):
    
    data = []

    for N in N_vals:
        print(f"\n=== Optimising with N = {N} layers ===")
        
        aqc_truncated = truncate_ansatz(aqc_ansatz, N)
        aqc_truncated.draw('mpl')

        aqc_truncated, aqc_params_truncated = truncate_parameters(aqc_ansatz, aqc_initial_parameters, N)
        # aqc_good_truncated = truncate_qiskit_circuit(aqc_good_circuit, N)
        # aqc_good_truncated.draw("mpl")

        objective = MaximizeUnitaryFidelity(aqc_target_mpo, aqc_truncated, simulator_settings)
        
        stopping_point = 1e-5
        
        # def my_loss_function(*args):
        #     val, grad = objective.loss_function(*args)
        #     print(f"Evaluating loss function: {1 - val:.8}")
        #     return val, grad
        
        def my_loss_function(x):
            val, grad = objective.loss_function(x)
            print(f"Evaluating fidelity: {1 - val:.8f}")
            return float(val), grad

        def callback(intermediate_result: OptimizeResult):
            print(f"Intermediate result: Fidelity {1 - intermediate_result.fun:.8}")
            if intermediate_result.fun < stopping_point:
                # Good enough for now
                raise StopIteration
            
        result = minimize(
            my_loss_function,
            aqc_params_truncated,
            # method="L-BFGS-B",
            method=adam,
            jac=False,
            options={"maxiter": 1000},
            callback=callback,
        )
        if result.status not in (
            0,
            1,
            99,
        ):  # 0 => success; 1 => max iterations reached; 99 => early termination via StopIteration
            raise RuntimeError(f"Optimisation failed: {result.message} (status={result.status})")

        print(f"Done after {result.nit} iterations.")
        aqc_final_parameters = result.x
        final_fidelity = 1 - result.fun

        # Store results
        data.append({
            "Trotter_steps": n_repetitions,
            "circuit_layers": N,
            "Final_fidelity": 1 - result.fun
        })

    print("\n\n=== Compression Results Table ===")
    print("Layers | Final Fidelity")
    for r in data:
        print(f"{r['circuit_layers']:>6} | {r['Final_fidelity']:.6f}")

    df = pd.DataFrame(data)

    return df, aqc_truncated, aqc_final_parameters

from scipy.optimize import OptimizeResult
import numpy as np

def adam(
    fun,
    x0,
    jac,
    args=(),
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    startiter=0,
    maxiter=100_000,
    callback=None,
    **kwargs,
):
    """``scipy.optimize.minimize`` compatible implementation of ADAM -
    [http://arxiv.org/pdf/1412.6980.pdf].
    Adapted from https://github.com/HIPS/autograd/blob/master/autograd/misc/optimizers.py.
    """
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for i in range(startiter, startiter + maxiter):
        val, g = fun(x)

        intermediate_result = OptimizeResult(
            x=x, fun=val, jac=g, nit=i, nfev=i, success=True, message="Intermediate result"
        )
        if callback is not None:
            try:
                callback(intermediate_result)
            except StopIteration:
                return OptimizeResult(
                    x=x, fun=val, jac=g, nit=i, nfev=i, success=True, status=99,
                    message="Desired cost reached early"
                )

        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1**(i + 1))  # bias correction.
        vhat = v / (1 - beta2**(i + 1))
        x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

    return OptimizeResult(
        x=x, fun=val, jac=g, nit=i + 1, nfev=i + 1, success=True, status=1,
        message="Max iterations reached"
    )


def compress_AQC_unitary_reverse(
    aqc_ansatz, aqc_initial_parameters, full_depth, aqc_target_mpo, simulator_settings, fidelity_threshold=0.98
):
    data = []
    prev_params = aqc_initial_parameters
    prev_ansatz = aqc_ansatz

    last_good_ansatz = None
    last_good_params = None


    for N in reversed(range(1, full_depth + 1)):
        print(f"\n=== Reverse Compression: Optimising with N = {N} layers ===")
        aqc_truncated = truncate_ansatz(prev_ansatz, N)
        aqc_truncated, aqc_params_truncated = truncate_parameters(aqc_truncated, prev_params, N)

        # Build correct mapping from prev_ansatz
        # full_param_binding = {
        #     param: val for param, val in zip(prev_ansatz.parameters, prev_params)
        # }

        # # Extract values for truncated circuit in correct order
        # aqc_params_truncated = [full_param_binding[p] for p in aqc_truncated.parameters]

        print(f"N = {N} → Truncated circuit has {aqc_truncated.num_parameters} parameters")

        # aqc_params_truncated = prev_params[:aqc_truncated.num_parameters]  
        assert len(aqc_params_truncated) == aqc_truncated.num_parameters
        
        print([g.operation for g in aqc_ansatz.data if g.operation.params])
        print([g.operation for g in aqc_truncated.data if g.operation.params])

        aqc_truncated = deepcopy(aqc_truncated)
        objective = MaximizeUnitaryFidelity(aqc_target_mpo, aqc_truncated, simulator_settings)
        
        stopping_point = 1e-5
        
        def my_loss_function(*args):
            val, grad = objective.loss_function(*args)
            print(f"Evaluating loss function: {1 - val:.8}")
            return val, grad
        
        # necessary for jax backend not explicit        
        # def my_loss_function(x):
        #     val, grad = objective.loss_function(x)
        #     if hasattr(val, "block_until_ready"):
        #         val = val.block_until_ready()
        #     val = float(val)

        #     print(f"Evaluating fidelity: {1 - float(val):.8f}")
        #     return val, grad
        
        def callback(intermediate_result: OptimizeResult):
            print(f"Intermediate result: Fidelity {1 - intermediate_result.fun:.8}")
            if intermediate_result.fun < stopping_point:
                # Good enough for now
                raise StopIteration

        result = minimize(
            my_loss_function,
            aqc_params_truncated,
            # method="L-BFGS-B",
            method = adam,
            jac=False,
            options={"maxiter": 1000},
            callback=callback,
        )

        if result.status not in (
            0,
            1,
            99,
        ):  # 0 => success; 1 => max iterations reached; 99 => early termination via StopIteration
            raise RuntimeError(f"Optimisation failed: {result.message} (status={result.status})")
        
        print(f"Done after {result.nit} iterations.")
        aqc_final_parameters = result.x
        final_fidelity = 1 - result.fun
        print(f"Final Fidelity at N={N}: {final_fidelity:.6f}")
        
        # Store results
        data.append({
            "circuit_layers": N,
            "Final_fidelity": final_fidelity
        })

        if final_fidelity >= fidelity_threshold:
            print(f"✅ Saving circuit with N = {N}, depth = {aqc_truncated.depth()}")
            last_good_ansatz = deepcopy(aqc_truncated)
            last_good_params = result.x
        else:
            print(f"\n❌ Fidelity below threshold at N={N}. Stopping.")
            break


        # Prepare for next loop
        prev_ansatz = aqc_truncated
        prev_params = result.x
        
    
    print("\n\n=== Compression Results Table ===")
    print("Layers | Final Fidelity")
    for r in data:
        print(f"{r['circuit_layers']:>6} | {r['Final_fidelity']:.6f}")
    
    df = pd.DataFrame(data)

    return df, last_good_ansatz, last_good_params


def compress_HS_AQC_unitary(
    gates_optimised, config, simulator_settings, aqc_target_mpo, N_vals
):
    data = []

    # Convert all brickwall gates to full circuit once
    gates_per_layer, _, _= get_gates_per_layer(
        gates_optimised,
        n_sites=config['n_sites'],
        degree=config['degree'],
        n_repetitions=config['n_repetitions'],
        hamiltonian=config['hamiltonian']
    )
    flat_gates = list(chain(*gates_per_layer))

    bw_qc = transform_to_bw_qc_unitarygate(
        num_sites=config['n_sites'],
        n_repetitions=config['n_repetitions'],
        system=config['hamiltonian'],
        gates=flat_gates
    )

    # Now decompose before generating the ansatz
    decomposed_circuit = decompose_unitary_gate(bw_qc)  # <-- this yields native gates
    aqc_ansatz_full, aqc_initial_parameters_full = generate_ansatz_from_circuit(decomposed_circuit)

    for N in N_vals:
        print(f"\n=== Optimizing with N = {N} native layers ===")

        # Truncate ansatz to N layers
        aqc_truncated = truncate_ansatz(aqc_ansatz_full, N)
        aqc_truncated, aqc_params_truncated = truncate_parameters(
            aqc_ansatz_full, aqc_initial_parameters_full, N
        )

        objective = MaximizeUnitaryFidelity(aqc_target_mpo, aqc_truncated, simulator_settings)
        stopping_point = 1e-5

        # def my_loss_function(*args):
        #     val, grad = objective.loss_function(*args)
        #     print(f"Evaluating loss function: {1 - val:.8}")
        #     return val, grad

        def my_loss_function(x):
            val, grad = objective.loss_function(x)
            print(f"Evaluating fidelity: {1 - val:.8f}")
            return float(val), grad

        def callback(intermediate_result: OptimizeResult):
            print(f"Intermediate result: Fidelity {1 - intermediate_result.fun:.8}")
            if intermediate_result.fun < stopping_point:
                raise StopIteration

        result = minimize(
            my_loss_function,
            aqc_params_truncated,
            # method="L-BFGS-B",
            method=adam,
            jac=False,
            options={"maxiter": 1000},
            callback=callback,
        )

        if result.status not in (0, 1, 99):
            raise RuntimeError(f"Optimization failed: {result.message} (status={result.status})")

        aqc_final_parameters = result.x
        final_fidelity = 1 - result.fun

        data.append({
            "Trotter_steps": config['n_repetitions'],
            "circuit_layers": N,
            "Final_fidelity": final_fidelity
        })

    print("\n\n=== Compression Results Table ===")
    print("Layers | Final Fidelity")
    for r in data:
        print(f"{r['circuit_layers']:>6} | {r['Final_fidelity']:.6f}")

    df = pd.DataFrame(data)
    return df, aqc_truncated, aqc_params_truncated, aqc_final_parameters


def compress_HS_AQC_unitary_reverse(
    gates_optimised, config, simulator_settings, full_depth, aqc_target_mpo, fidelity_threshold=0.98
):
    data = []

    # Convert all brickwall gates to full circuit once
    gates_per_layer, _, _= get_gates_per_layer(
        gates_optimised,
        n_sites=config['n_sites'],
        degree=config['degree'],
        n_repetitions=config['n_repetitions'],
        hamiltonian=config['hamiltonian']
    )
    flat_gates = list(chain(*gates_per_layer))

    bw_qc = transform_to_bw_qc_unitarygate(
        num_sites=config['n_sites'],
        n_repetitions=config['n_repetitions'],
        system=config['hamiltonian'],
        gates=flat_gates
    )

    # Now decompose before generating the ansatz
    decomposed_circuit = decompose_unitary_gate(bw_qc)  # <-- this yields native gates
    aqc_ansatz_full, aqc_initial_parameters_full = generate_ansatz_from_circuit(decomposed_circuit)

    prev_params = aqc_initial_parameters_full
    prev_ansatz = aqc_ansatz_full

    last_good_ansatz = None
    last_good_params = None

    for N in reversed(range(1, full_depth + 1)):
        print(f"\n=== Optimizing with N = {N} native layers ===")

        # Truncate ansatz to N layers
        aqc_truncated = truncate_ansatz(prev_ansatz, N)
        aqc_truncated, aqc_params_truncated = truncate_parameters(
            aqc_truncated, prev_params, N
        )

        print(f"N = {N} → Truncated circuit has {aqc_truncated.num_parameters} parameters")

        assert len(aqc_params_truncated) == aqc_truncated.num_parameters
        
        aqc_truncated = deepcopy(aqc_truncated)
        objective = MaximizeUnitaryFidelity(aqc_target_mpo, aqc_truncated, simulator_settings)
        
        stopping_point = 1e-5

        def my_loss_function(*args):
            val, grad = objective.loss_function(*args)
            print(f"Evaluating loss function: {1 - val:.8}")
            return val, grad

        # jax backedn
        # def my_loss_function(x):
        #     val, grad = objective.loss_function(x)
        #     if hasattr(val, "block_until_ready"):
        #         val = val.block_until_ready()
        #     val = float(val)

        #     print(f"Evaluating fidelity: {1 - float(val):.8f}")
        #     return val, grad

        def callback(intermediate_result: OptimizeResult):
            print(f"Intermediate result: Fidelity {1 - intermediate_result.fun:.8}")
            if intermediate_result.fun < stopping_point:
                raise StopIteration

        result = minimize(
            my_loss_function,
            aqc_params_truncated,
            # method="L-BFGS-B",
            method = adam,
            jac=False,
            options={"maxiter": 1000},
            callback=callback,
        )

        if result.status not in (0, 1, 99):
            raise RuntimeError(f"Optimization failed: {result.message} (status={result.status})")

        print(f"Done after {result.nit} iterations.")
        aqc_final_parameters = result.x
        final_fidelity = 1 - result.fun
        print(f"Final Fidelity at N={N}: {final_fidelity:.6f}")
        

        data.append({
            "Trotter_steps": config['n_repetitions'],
            "circuit_layers": N,
            "Final_fidelity": final_fidelity
        })

        if final_fidelity >= fidelity_threshold:
            print(f"✅ Saving circuit with N = {N}, depth = {aqc_truncated.depth()}")
            last_good_ansatz = deepcopy(aqc_truncated)
            last_good_params = result.x
        else:
            print(f"\n❌ Fidelity below threshold at N={N}. Stopping.")
            break

        # Prepare for next loop
        prev_ansatz = aqc_truncated
        prev_params = result.x

    print("\n\n=== Compression Results Table ===")
    print("Layers | Final Fidelity")
    for r in data:
        print(f"{r['circuit_layers']:>6} | {r['Final_fidelity']:.6f}")

    df = pd.DataFrame(data)

    return df, last_good_ansatz, last_good_params


def plot_methods_comparison(df_dict, x_col='circuit_layers', y_col='Final_fidelity',
                                     x_label='Circuit Layers', y_label='Fidelity',
                                     title='Method Comparison', save_path=None):
    """
    Plot multiple methods by extracting data from DataFrames.

    Parameters:
    - df_dict: Dictionary where keys are method names and values are DataFrames.
    - x_col: Name of the x-axis column in each DataFrame.
    - y_col: Name of the y-axis column in each DataFrame.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title of the plot.
    - save_path: If provided, saves the plot to this path.
    """
    plt.style.use('/Users/aag/Documents/ropt-aqc/molecular.mplstyle')
    plt.figure(figsize=(11, 10))

    color_list = plt.cm.get_cmap('tab10', len(df_dict))  # tab10 or Set1 are good options
    
    for i, (method_name, df) in enumerate(df_dict.items()):
        plt.plot(df[x_col], df[y_col],
                 marker='o',
                 label=method_name,
                 linewidth=1.5,
                 markersize=8,
                 color=color_list(i))  # Assign a unique color

    plt.xlabel(x_label, fontsize=26)
    plt.ylabel(y_label, fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title(title)
    plt.legend(fontsize=20, loc='lower right')
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax = plt.gca()

    ax.text(
        0.95, 0.05,    # x, y in axes‐coords
        "(a)",         # the label you want
        transform=ax.transAxes,
        fontsize=28,
        va="top",      # vertical alignment
        ha="left"      # horizontal alignment
    )

    if save_path:
        fname = title + ".pdf"
        fdir = os.path.join(os.getcwd(), "Figures", fname)
        plt.savefig(fdir, bbox_inches='tight', dpi=300)
    plt.show()


def compress_AQC_unitary_with_checkpoint(aqc_ansatz, aqc_initial_parameters, aqc_target_mpo, N_vals, simulator_settings, n_repetitions, checkpoint_path=None):
    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
            data = checkpoint["data"]
            completed_N = set(r["circuit_layers"] for r in data)
            print(f"✅ Loaded checkpoint. Already completed: {sorted(completed_N)}")
    else:
        data = []
        completed_N = set()

    last_good_ansatz = None
    last_good_params = None

    for N in N_vals:
        if N in completed_N:
            print(f"⏭️ Skipping N = {N}, already in checkpoint")
            continue

        print(f"\n=== Optimizing with N = {N} layers ===")

        # Truncate circuit and parameters
        aqc_truncated = truncate_ansatz(aqc_ansatz, N)
        aqc_truncated, aqc_params_truncated = truncate_parameters(aqc_ansatz, aqc_initial_parameters, N)

        if len(aqc_params_truncated) == 0:
            print("⚠️ Skipping optimization: no parameters used at this depth.")
            final_fidelity = None
            data.append({
                "Trotter_steps": n_repetitions,
                "circuit_layers": N,
                "Final_fidelity": final_fidelity,
            })
            continue

        objective = MaximizeUnitaryFidelity(aqc_target_mpo, aqc_truncated, simulator_settings)
        stopping_point = 1e-5

        def my_loss_function(x):
            val, grad = objective.loss_function(x)
            print(f"Evaluating fidelity: {1 - val:.8f}")
            return float(val), grad

        def callback(intermediate_result: OptimizeResult):
            print(f"Intermediate result: Fidelity {1 - intermediate_result.fun:.8}")
            if intermediate_result.fun < stopping_point:
                raise StopIteration

        try:
            result = minimize(
                my_loss_function,
                aqc_params_truncated,
                method=adam,
                jac=False,
                options={"maxiter": 1000},
                callback=callback,
            )

            if result.status not in (0, 1, 99):
                raise RuntimeError(f"Optimization failed: {result.message} (status={result.status})")

            print(f"✅ Done after {result.nit} iterations.")
            aqc_final_parameters = result.x
            final_fidelity = 1 - result.fun

            last_good_ansatz = aqc_truncated
            last_good_params = aqc_final_parameters

        except Exception as e:
            print(f"❌ Optimization failed for N={N}: {e}")
            final_fidelity = None
            aqc_final_parameters = None

        # Save progress
        data.append({
            "Trotter_steps": n_repetitions,
            "circuit_layers": N,
            "Final_fidelity": final_fidelity,
        })

        if checkpoint_path:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "data": data,
                    "last_ansatz": last_good_ansatz,
                    "last_params": last_good_params,
                }, f)

    print("\n\n=== Compression Results Table ===")
    print("Layers | Final Fidelity")
    for r in data:
        print(f"{r['circuit_layers']:>6} | {r['Final_fidelity'] if r['Final_fidelity'] is not None else 'N/A'}")

    df = pd.DataFrame(data)
    return df, last_good_ansatz, last_good_params


def compress_AQC_unitary_reverse_with_checkpoint(
    aqc_ansatz, aqc_initial_parameters, full_depth, aqc_target_mpo,
    simulator_settings, fidelity_threshold=0.98, checkpoint_path=None
):
    # Initialize or load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"🔄 Loaded checkpoint from {checkpoint_path}")
        data = checkpoint["data"]
        completed_N = set(d["circuit_layers"] for d in data)
        prev_params = checkpoint["prev_params"]
        prev_ansatz = checkpoint["prev_ansatz"]
        last_good_ansatz = checkpoint["last_good_ansatz"]
        last_good_params = checkpoint["last_good_params"]
    else:
        data = []
        completed_N = set()
        prev_params = aqc_initial_parameters
        prev_ansatz = aqc_ansatz
        last_good_ansatz = None
        last_good_params = None

    for N in reversed(range(1, full_depth + 1)):
        if N in completed_N:
            print(f"⏭️  Skipping already completed N = {N}")
            continue

        print(f"\n=== Reverse Compression: Optimising with N = {N} layers ===")

        aqc_truncated = truncate_ansatz(prev_ansatz, N)
        aqc_truncated, aqc_params_truncated = truncate_parameters(aqc_truncated, prev_params, N)

        assert len(aqc_params_truncated) == aqc_truncated.num_parameters

        aqc_truncated = deepcopy(aqc_truncated)
        objective = MaximizeUnitaryFidelity(aqc_target_mpo, aqc_truncated, simulator_settings)
        stopping_point = 1e-5

        def my_loss_function(*args):
            val, grad = objective.loss_function(*args)
            print(f"Evaluating loss function: {1 - val:.8}")
            return val, grad

        def callback(intermediate_result: OptimizeResult):
            print(f"Intermediate result: Fidelity {1 - intermediate_result.fun:.8}")
            if intermediate_result.fun < stopping_point:
                raise StopIteration

        result = minimize(
            my_loss_function,
            aqc_params_truncated,
            method=adam,
            jac=False,
            options={"maxiter": 1000},
            callback=callback,
        )

        if result.status not in (0, 1, 99):
            raise RuntimeError(f"Optimisation failed: {result.message} (status={result.status})")

        print(f"✅ Done after {result.nit} iterations.")
        final_fidelity = 1 - result.fun

        data.append({
            "circuit_layers": N,
            "Final_fidelity": final_fidelity
        })

        if final_fidelity >= fidelity_threshold:
            print(f"✅ Saving circuit with N = {N}, depth = {aqc_truncated.depth()}")
            last_good_ansatz = deepcopy(aqc_truncated)
            last_good_params = result.x
        else:
            print(f"❌ Fidelity below threshold at N={N}. Stopping.")
            break

        prev_ansatz = aqc_truncated
        prev_params = result.x

        if checkpoint_path:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    "data": data,
                    "prev_params": prev_params,
                    "prev_ansatz": prev_ansatz,
                    "last_good_ansatz": last_good_ansatz,
                    "last_good_params": last_good_params
                }, f)
            print(f"💾 Checkpoint saved at N={N} to {checkpoint_path}")

    print("\n\n=== Compression Results Table ===")
    print("Layers | Final Fidelity")
    for r in data:
        print(f"{r['circuit_layers']:>6} | {r['Final_fidelity']:.6f}")

    df = pd.DataFrame(data)
    return df, last_good_ansatz, last_good_params


def compress_HS_AQC_unitary_with_checkpoint(
    gates_optimised, config, simulator_settings, aqc_target_mpo, N_vals, checkpoint_path=None
):
    data = []
    completed_N = set()
    best_fidelity = -1
    best_result = (None, None, None)

    # Try to load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
            completed_N = set(checkpoint["completed_N"])
            data = checkpoint["data"]
            best_result = checkpoint["best_result"]
            best_fidelity = checkpoint["best_fidelity"]

    # Convert brickwall gates to full decomposed ansatz
    gates_per_layer, _, _ = get_gates_per_layer(
        gates_optimised,
        n_sites=config["n_sites"],
        degree=config["degree"],
        n_repetitions=config["n_repetitions"],
        hamiltonian=config["hamiltonian"],
    )
    flat_gates = list(chain(*gates_per_layer))

    bw_qc = transform_to_bw_qc_unitarygate(
        num_sites=config["n_sites"],
        n_repetitions=config["n_repetitions"],
        system=config["hamiltonian"],
        gates=flat_gates,
    )
    decomposed_circuit = decompose_unitary_gate(bw_qc)
    aqc_ansatz_full, aqc_initial_parameters_full = generate_ansatz_from_circuit(decomposed_circuit)

    for N in N_vals:
        if N in completed_N:
            continue

        print(f"\n=== Optimizing with N = {N} native layers ===")

        aqc_truncated = truncate_ansatz(aqc_ansatz_full, N)
        aqc_truncated, aqc_params_truncated = truncate_parameters(
            aqc_ansatz_full, aqc_initial_parameters_full, N
        )

        objective = MaximizeUnitaryFidelity(aqc_target_mpo, aqc_truncated, simulator_settings)
        stopping_point = 1e-5

        def my_loss_function(x):
            val, grad = objective.loss_function(x)
            print(f"Evaluating fidelity: {1 - val:.8f}")
            return float(val), grad

        def callback(intermediate_result):
            print(f"Intermediate result: Fidelity {1 - intermediate_result.fun:.8}")
            if intermediate_result.fun < stopping_point:
                raise StopIteration

        result = minimize(
            my_loss_function,
            aqc_params_truncated,
            method=adam,
            jac=False,
            options={"maxiter": 1000},
            callback=callback,
        )

        if result.status not in (0, 1, 99):
            raise RuntimeError(f"Optimization failed: {result.message} (status={result.status})")

        aqc_final_parameters = result.x
        final_fidelity = 1 - result.fun

        row = {
            "Trotter_steps": config["n_repetitions"],
            "circuit_layers": N,
            "Final_fidelity": final_fidelity,
        }
        data.append(row)
        completed_N.add(N)

        if final_fidelity > best_fidelity:
            best_fidelity = final_fidelity
            best_result = (deepcopy(aqc_truncated), aqc_params_truncated, aqc_final_parameters)

        # Save checkpoint
        if checkpoint_path:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "data": data,
                    "completed_N": list(completed_N),
                    "best_result": best_result,
                    "best_fidelity": best_fidelity
                }, f)

    print("\n\n=== Compression Results Table ===")
    print("Layers | Final Fidelity")
    for r in data:
        print(f"{r['circuit_layers']:>6} | {r['Final_fidelity']:.6f}")

    df = pd.DataFrame(data)
    aqc_truncated, aqc_params_truncated, aqc_final_parameters = best_result
    return df, aqc_truncated, aqc_params_truncated, aqc_final_parameters


def compress_HS_AQC_unitary_reverse_checkpointed(
    gates_optimised, config, simulator_settings, full_depth, aqc_target_mpo,
    fidelity_threshold=0.98, checkpoint_path=None
):
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"🧠 Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        data = checkpoint["data"]
        prev_params = checkpoint["prev_params"]
        prev_ansatz = checkpoint["prev_ansatz"]
        N_resume = checkpoint["N"]
        last_good_ansatz = checkpoint["last_good_ansatz"]
        last_good_params = checkpoint["last_good_params"]
    else:
        data = []

        gates_per_layer, _, _ = get_gates_per_layer(
            gates_optimised,
            n_sites=config['n_sites'],
            degree=config['degree'],
            n_repetitions=config['n_repetitions'],
            hamiltonian=config['hamiltonian']
        )
        flat_gates = list(chain(*gates_per_layer))

        bw_qc = transform_to_bw_qc_unitarygate(
            num_sites=config['n_sites'],
            n_repetitions=config['n_repetitions'],
            system=config['hamiltonian'],
            gates=flat_gates
        )

        decomposed_circuit = decompose_unitary_gate(bw_qc)
        aqc_ansatz_full, aqc_initial_parameters_full = generate_ansatz_from_circuit(decomposed_circuit)

        prev_params = aqc_initial_parameters_full
        prev_ansatz = aqc_ansatz_full
        last_good_ansatz = None
        last_good_params = None
        N_resume = full_depth

    for N in reversed(range(1, N_resume + 1)):
        print(f"\n=== Optimizing with N = {N} native layers ===")

        aqc_truncated = truncate_ansatz(prev_ansatz, N)
        aqc_truncated, aqc_params_truncated = truncate_parameters(aqc_truncated, prev_params, N)

        aqc_truncated = deepcopy(aqc_truncated)
        objective = MaximizeUnitaryFidelity(aqc_target_mpo, aqc_truncated, simulator_settings)

        stopping_point = 1e-5

        def my_loss_function(*args):
            val, grad = objective.loss_function(*args)
            print(f"Evaluating loss function: {1 - val:.8}")
            return val, grad

        def callback(intermediate_result: OptimizeResult):
            print(f"Intermediate result: Fidelity {1 - intermediate_result.fun:.8}")
            if intermediate_result.fun < stopping_point:
                raise StopIteration

        result = minimize(
            my_loss_function,
            aqc_params_truncated,
            method="L-BFGS-B",
            jac=False,
            options={"maxiter": 1000},
            callback=callback,
        )

        if result.status not in (0, 1, 99):
            raise RuntimeError(f"Optimization failed: {result.message} (status={result.status})")

        aqc_final_parameters = result.x
        final_fidelity = 1 - result.fun

        print(f"Final Fidelity at N={N}: {final_fidelity:.6f}")

        data.append({
            "Trotter_steps": config['n_repetitions'],
            "circuit_layers": N,
            "Final_fidelity": final_fidelity
        })

        if final_fidelity >= fidelity_threshold:
            print(f"✅ Saving circuit with N = {N}, depth = {aqc_truncated.depth()}")
            last_good_ansatz = deepcopy(aqc_truncated)
            last_good_params = result.x
        else:
            print(f"\n❌ Fidelity below threshold at N={N}. Stopping.")
            break

        prev_ansatz = aqc_truncated
        prev_params = result.x

        # 🔁 Save checkpoint
        if checkpoint_path:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "data": data,
                    "prev_params": prev_params,
                    "prev_ansatz": prev_ansatz,
                    "last_good_ansatz": last_good_ansatz,
                    "last_good_params": last_good_params,
                    "N": N - 1
                }, f)

    df = pd.DataFrame(data)
    return df, last_good_ansatz, last_good_params
