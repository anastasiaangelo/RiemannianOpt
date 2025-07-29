import numpy as np
import quimb as qu
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from mpo_lib.models.pxp_model import PXPModel1D
from mpo_lib.circuit.brickwall import BrickwallCircuit

from ropt_aqc.circuit_building import remap_layered_gate_indices

import qiskit_quimb
from qiskit import transpile, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit_addon_utils.problem_generators import generate_time_evolution_circuit

from rustworkx.generators import path_graph

def test_magnetisation_evolution(binary, num_sites, num_steps, time_steps):
    state = qtn.CircuitMPS(N=num_sites)

    for idx, b in enumerate(binary):
        if b == '1':
            state.apply_gate('X', qubits=(idx,))

    circuit = PXPModel1D.first_order(num_sites=num_sites, final_time=1, num_steps=num_steps)
    circuit.combine_layers()

    magnetisation = []
    for _ in tqdm(np.arange(time_steps)):
        for layer in tqdm(circuit.layers, leave=False):
            for unitary in layer.unitaries:
                if unitary.array.shape == (2, 2, 2, 2):
                    state.apply_gate_raw(unitary.array.transpose(2, 3, 0, 1), where=unitary.sites)
                else:
                    state.apply_gate_raw(unitary.array.transpose(1, 0), where=unitary.sites)

        for site in range(num_sites):
            magnetisation.append(state.local_expectation(qu.pauli('Z'), where=site))

    magnetisation = np.real(magnetisation).reshape(time_steps, num_sites)
    occupation = np.zeros((time_steps, num_sites-1))
    for i in range(num_sites-1):
        occupation[:, i] = (-1)**i * 0.5 * (magnetisation[:, i] + magnetisation[:, i+1])

    plt.figure(figsize=(6.4, 4.8))

    plt.pcolor(
        np.arange(occupation.shape[1]) + 0.5,
        np.arange(time_steps),
        occupation,
        cmap='bwr', vmin=-1.0, vmax=1.0, shading='nearest'
    )

    plt.show()

    return magnetisation


def test_magnetisation_qiskit(num_sites, binary, time_steps, layered_gates_remapped=None, qc=None):
    """
    Simulates the magnetisation evolution of a PXP model using either a Qiskit circuit or a list of remapped gate layers.
    
    Args:
        num_sites (int): Number of sites in the PXP model.
        binary (str): Binary string representing the initial state of the system.
        time_steps (int): Number of time steps to simulate.
        layered_gates_remapped (list, optional): List of remapped gate layers. If None, `qc` must be provided.
        qc (QuantumCircuit, optional): Qiskit circuit to simulate. If None, `layered_gates_remapped` must be provided.
    Returns:
        np.ndarray: Magnetisation values at each time step.
    """
    if layered_gates_remapped is None:
        if qc is None:
            raise ValueError("Either 'layered_gates_remapped' or 'qc' must be provided.")
        layered_gates_remapped = convert_qiskit_layers_to_gate_tensors(qc)

    state = qtn.CircuitMPS(num_sites)

    for idx, b in enumerate(binary):
        if b == '1':
            state.apply_gate('X', qubits=(idx,))

    magnetisation = []

    for _ in tqdm(range(time_steps)):
        for layer in layered_gates_remapped:
            for gate, sites, *_ in layer:
                shape = gate.shape
                if shape == (2, 2, 2, 2):
                    gate_tensor = gate.transpose(2, 3, 0, 1)
                elif shape == (4, 4):
                    gate_tensor = gate.reshape(2, 2, 2, 2).transpose(2, 3, 0, 1)
                elif shape == (2, 2):
                    gate_tensor = gate.T
                else:
                    raise ValueError(f"Unsupported gate shape: {shape}")
                
                state.apply_gate_raw(gate_tensor, where=sites)

        for site in range(num_sites):
            magnetisation.append(state.local_expectation(qu.pauli('Z'), where=site))

    magnetisation = np.real(magnetisation).reshape(time_steps, num_sites)
    occupation = np.zeros((time_steps, num_sites-1))
    for i in range(num_sites-1):
        occupation[:, i] = (-1)**i * 0.5 * (magnetisation[:, i] + magnetisation[:, i+1])

    plt.figure(figsize=(6.4, 4.8))

    plt.pcolor(
    np.arange(occupation.shape[1]),
    np.arange(time_steps),
    occupation,
    cmap='bwr', vmin=-1.0, vmax=1.0, shading='nearest'
    )

    plt.show()

    return magnetisation

def quimb_to_layered_gate_list(quimb_model) -> list[list[tuple[np.ndarray, list[int]]]]:
    """
    Extracts the circuit as a list of gate layers.
    Each layer is a list of (gate_matrix, site_indices) tuples.

    Returns:
        List of layers, where each layer is a list of tuples (unitary array, sites).
    """
    layered = []
    for layer in quimb_model.layers:
        gates = []
        for unitary in layer.unitaries:
            arr = np.array(unitary.array, dtype=np.complex128)  # precision-safe
            gates.append((arr.copy(), unitary.sites))  # ensure no mutation
        layered.append(gates)
    
    return layered


def name_layered_gates(layered_gates):
    named_layered_gates = []

    for layer_idx, layer in enumerate(layered_gates):
        named_layer = []
        for gate_idx, (gate_tensor, qubits) in enumerate(layer):
            name = f"G_L{layer_idx}_G{gate_idx}"
            named_layer.append((gate_tensor, qubits, name))
            # print(f"{name} -> qubits {qubits}")
        named_layered_gates.append(named_layer)

    return named_layered_gates


def merge_single_qubit_gates(named_layered_gates, with_names=True, use_TN=False):
    merged_layers = []
    pending_1q = {}

    for layer_idx in reversed(range(len(named_layered_gates))):
        layer = named_layered_gates[layer_idx]
        new_layer = []

        for gate_idx, entry in enumerate(layer):
            gate, qubits, *maybe_name = entry
            name = maybe_name[0] if maybe_name else f"G_L{layer_idx}_G{gate_idx}"

            if len(qubits) == 1:
                pending_1q[qubits[0]] = (gate, name)
            elif len(qubits) == 2:
                q0, q1 = qubits
                G = gate.reshape(4, 4)
                merged_name = name

                if q0 in pending_1q:
                    U, u_name = pending_1q.pop(q0)
                    G = G @ np.kron(U, np.eye(2))
                    merged_name += f"_m_{u_name}"

                if q1 in pending_1q:
                    U, u_name = pending_1q.pop(q1)
                    G = G @ np.kron(np.eye(2), U)
                    merged_name += f"_m_{u_name}"

                G_tensor = G.reshape(2, 2, 2, 2)
                if with_names:
                    new_layer.append((G_tensor, [q0, q1], merged_name))
                else:
                    new_layer.append((G_tensor, [q0, q1]))
            else:
                raise ValueError("Unsupported gate arity")

        merged_layers.insert(0, new_layer)

    if use_TN:
        return np.stack([gate for layer in merged_layers for gate, *_ in layer])

    return merged_layers


def get_brickwall_twoqubit_layered_gates(t, n_sites, n_repetitions, use_TN=True, qubit_direction="left-to-right"):
    
    model_first_order = PXPModel1D.first_order(n_sites, t, n_repetitions)
    layered_gates = BrickwallCircuit.to_layered_gate_list(model_first_order)
    remapped_layers = remap_layered_gate_indices(layered_gates, n_sites)
    merged_layers = merge_single_qubit_gates(remapped_layers, with_names=False, use_TN=use_TN)

    if use_TN:
        return merged_layers
    else:
        return [gate.reshape(4, 4) for layer in merged_layers for gate, _ in layer]


def extract_layers_from_circuit(qc):
    """
    Extracts layers of gates from a Qiskit circuit and returns them as a list of gate layers along with their qubit indices.
    
    Returns:
        List of layers, where each layer is a list of tuples (gate, qubit_indices).
        Each tuple contains the gate object and a list of qubit indices it acts on.
    """
    dag = circuit_to_dag(qc)
    layers = list(dag.layers())

    gate_layers = []
    for layer in layers:
        operations = layer['graph'].op_nodes()
        gate_list = []
        for op_node in operations:
            if op_node.name != 'barrier':
                qubit_indices = [qc.find_bit(q).index for q in op_node.qargs]
                gate_list.append((op_node.op, qubit_indices))
        if gate_list:
            gate_layers.append(gate_list)
    return gate_layers


def convert_qiskit_layers_to_gate_tensors(qc):
    """
    Converts a Qiskit circuit into a list of gate tensors, where each tensor represents a gate in the circuit.
    Converts each gate into a tensor format for simulations.
    
    Returns:    
        List of tensorised gate layers, where each layer is a list of tuples (tensor, qubit_indices, gate_name).
    """
    gate_layers = extract_layers_from_circuit(qc)
    layered_gates_remapped = []

    for layer in gate_layers:
        layer_tensor_list = []
        for gate, qubit_indices in layer:
            matrix = Operator(gate).data  # Get unitary matrix as numpy array

            # Determine tensor shape
            shape = matrix.shape
            if shape == (4, 4):  # 2-qubit gate
                tensor = matrix.reshape(2, 2, 2, 2).transpose(2, 3, 0, 1)
            elif shape == (2, 2):  # 1-qubit gate
                tensor = matrix.T  # qtn expects gates in (out, in)
            else:
                raise ValueError(f"Unsupported gate shape {shape}")
            
            layer_tensor_list.append((tensor, qubit_indices, gate.name))
        layered_gates_remapped.append(layer_tensor_list)

    return layered_gates_remapped


def pxp_hamiltonian_sparse(num_qubits):
    terms = []
    coeffs = []

    for i in range(num_qubits):
        left = i - 1
        right = i + 1

        def padded(pauli_str):
            s = ['I'] * num_qubits
            for j, p in zip([left, i, right], pauli_str):
                if 0 <= j < num_qubits:
                    s[j] = p
            return ''.join(s)

        # Add the 4 terms from expansion of P X P = ¼ (X + Z X + X Z + Z X Z)
        terms.extend([
            padded("IXI"),   # just X_i
            padded("ZXI"),   # Z_{i-1} X_i
            padded("IXZ"),   # X_i Z_{i+1}
            padded("ZXZ")    # Z_{i-1} X_i Z_{i+1}
        ])
        coeffs.extend([0.25, 0.25, 0.25, 0.25])

    return SparsePauliOp.from_list(list(zip(terms, coeffs)))


def generate_reference_mpo(aqc_evolution_time, aqc_target_log2_num_trotter_steps, hamiltonian):
    """
    Generate a high fidelity reference MPO with many Trotter steps to approximate the true time evolution
    aqc_target_log2_num_trotter_steps should be larger than the number of Trotter steps of the circuit to be optimised
    aqc_evolution_time should match the evolution time of the system being optimised
    """
    aqc_target_step_circuit = generate_time_evolution_circuit(
        hamiltonian,
        synthesis=SuzukiTrotter(reps=1),
        time=aqc_evolution_time / 2**aqc_target_log2_num_trotter_steps,
    )

    print(
        f"This single Trotter step circuit will be repeated {2**aqc_target_log2_num_trotter_steps} times:"
    )
    aqc_target_step_circuit.draw("mpl", fold=-1)

    aqc_target_step_circuit = transpile(
        aqc_target_step_circuit,
        basis_gates=["rx", "rz", "cx"]
    )
    gates = qiskit_quimb.quimb_gates(aqc_target_step_circuit)
    target_circ = qtn.Circuit.from_gates(
        gates,
        gate_contract="split-gate",
        tag_gate_numbers=False,
    )

    target_circ_mps = qtn.CircuitMPS.from_gates(
        gates,
        gate_contract="split-gate",
        tag_gate_numbers=False,
    )
    tn_uni = target_circ.get_uni()
    tn_uni.draw(tn_uni.site_tags, show_tags=False)

    # compress via fitting:
    cutoff = 1e-8

    aqc_target_mpo = qtn.tensor_network_1d_compress(
        tn_uni,
        max_bond=32,
        cutoff=cutoff,
        method="fit",
        bsz=2,  # bsz=1 is cheaper per sweep, but possibly slower to converge
        max_iterations=100,
        tol=1e-6,
        progbar=True,
    )
    aqc_target_mpo.distance_normalized(tn_uni)

    # cast as MPO 
    aqc_target_mpo.view_as_(
        qtn.MatrixProductOperator,
        cyclic=False,
        L=target_circ.N,
    )
    
    # repeatedly sqaure and compress MPO to represent an operator with many Trotter steps
    for _ in range(aqc_target_log2_num_trotter_steps):
        aqc_target_mpo = aqc_target_mpo.apply(aqc_target_mpo)
        aqc_target_mpo.compress(cutoff=cutoff, max_bond=64)
        aqc_target_mpo.show()
    aqc_target_mpo

    return aqc_target_mpo, aqc_target_step_circuit, target_circ_mps


def calcualte_initial_final_Frobenius(num_sites, aqc_target_step_circuit, aqc_target_log2_num_trotter_steps, optimised_qc, initial_qc=None):
    coupling_map = path_graph(num_sites)
    initial_fidelity = None

    if coupling_map.num_nodes() <= 8:
        target_operator = Operator(aqc_target_step_circuit).power(2**aqc_target_log2_num_trotter_steps)
        
        if initial_qc:
            initial_fidelity = abs(
                np.trace(Operator(initial_qc).conjugate().to_matrix() @ target_operator.to_matrix())
                / 2.0 ** coupling_map.num_nodes()
            )
            print(f"Initial Frobenius inner product: {initial_fidelity:.8}")

        final_fidelity = abs(
            np.trace(
                Operator(optimised_qc).conjugate().to_matrix() @ target_operator.to_matrix()
            )
            / 2.0 ** coupling_map.num_nodes()
        )
        print(f"Final Frobenius inner product: {final_fidelity:.8}")

    return initial_fidelity, final_fidelity


def plot_fidelity_decay_multiple_compressed_circuits(
    compressed_circuit_dict,
    reference_step_circuit,
    final_time,
    num_sites,
    trotter_log2_steps,
    max_steps=20
):
    time_points = []
    fidelity_dict = {label: [] for label in compressed_circuit_dict}

    num_steps_list = range(1, max_steps + 1)

    for n in num_steps_list:
        total_time = n * final_time
        time_points.append(total_time)

        # Compute reference unitary for n steps
        reference_n = Operator(reference_step_circuit).power(n)

        for label, compressed_step in compressed_circuit_dict.items():
            # Compose compressed circuit n times
            qc_compressed = QuantumCircuit(num_sites)
            for _ in range(n):
                qc_compressed = qc_compressed.compose(compressed_step)

            # Check circuit equivalence
            U_single = Operator(compressed_step).data
            U_expected = np.linalg.matrix_power(U_single, n)
            U_actual = Operator(qc_compressed).data
            assert np.allclose(U_expected, U_actual, atol=1e-6), f"❌ Mismatch in circuit {label}, step {n}"

            # Compute fidelity
            _, fidelity = calcualte_initial_final_Frobenius(
                num_sites,
                reference_n,
                trotter_log2_steps,
                qc_compressed
            )
            fidelity_dict[label].append(fidelity)
            print(f"✅ {label}, step {n}, fidelity = {fidelity:.8f}")

    # Plotting
    plt.figure(figsize=(7, 5))
    for label, fids in fidelity_dict.items():
        plt.plot(time_points, fids, 'o-', label=label)

    plt.xlabel("Total time composed circuit")
    plt.ylabel("Fidelity")
    plt.ylim(0.65, 1.01)
    plt.grid(True)
    plt.title("Fidelity Decay of Compressed Circuits Over Time")
    plt.legend()
    plt.savefig("/Users/aag/Documents/ropt-aqc/PXP/Figures/fidelity_decay_multiple_compressed_circuits.pdf")
    plt.show()

    return time_points, fidelity_dict


def mpo_to_gate_tensor_list(mpo):
    """Convert a quimb MPO into a list of (Dl, Dr, 2, 2) numpy arrays."""
    gate_list = []

    for i, tensor in enumerate(mpo):
        arr = tensor.data
        shape = arr.shape

        if shape[-2:] != (2, 2):
            raise ValueError(f"Expected last two dims to be (2,2), got {shape}")

        phys_shape = shape[-2:]
        bond_shape = shape[:-2]

        # Determine Dl and Dr
        if len(bond_shape) == 0:
            Dl, Dr = 1, 1
        elif len(bond_shape) == 1:
            Dl, Dr = (1, bond_shape[0]) if i == 0 else (bond_shape[0], 1)
        elif len(bond_shape) == 2:
            Dl, Dr = bond_shape
        else:
            raise ValueError(f"Too many bond dimensions in tensor shape {shape}")

        gate = arr.reshape(Dl, Dr, 2, 2)
        gate_list.append(gate)

    return gate_list


import jax.numpy as jnp
from scipy.linalg import expm

# Define Pauli matrices and CZ
I = jnp.eye(2)
X = jnp.asarray([[0., 1.],[1., 0.]])
Z = jnp.asarray([[1., 0.],[0.,-1.]])


XI = jnp.kron(X, I)
IX = jnp.kron(I, X)
Z_op = Z
CZ = jnp.diag(jnp.array([1, 1, 1, -1]))


def pxp_interaction_hamiltonian(rx_left=True):
    if rx_left:
        return 0.5 * XI + CZ
    else:
        return 0.5 * IX + CZ


def pxp_potential_hamiltonian(site_idx: int, mu: float, chi: float):
    coeff = mu + ((-1) ** site_idx) * (chi / 2)
    return 0.5 * coeff * Z_op

from scipy.linalg import expm
import numpy as np
import jax.numpy as jnp
def get_brickwall_trotter_gates_spin_chain_pxp(
    t,
    n_sites,
    n_repetitions=1,
    degree=1,
    use_TN=False,
    return_format='layered',  # 'layered' or 'flat'
    omega=0.5,
    mu=-3.0,
    chi=0.0,
):
    dt = t / n_repetitions
    if degree == 2:
        dt /= 2

    even_pairs = [[i, i + 1] for i in range(0, n_sites - 1, 2)]
    odd_pairs  = [[i, i + 1] for i in range(1, n_sites - 1, 2)]

    def make_interaction_gates(pairs, rx_left):
        gate_list = []
        for pair in pairs:
            H = pxp_interaction_hamiltonian(rx_left=rx_left)
            theta = omega * dt
            if rx_left and pair[0] == 0:
                theta *= 2
            elif not rx_left and pair[1] == n_sites - 1:
                theta *= 2
            gate = expm(-1j * theta * H).reshape(2, 2, 2, 2)
            gate_list.append((gate, tuple(np.int64(x) for x in pair))) 
        return gate_list

    def make_potential_gates():
        return [(expm(-1j * dt * pxp_potential_hamiltonian(j, mu, chi)).reshape(2, 2), (np.int64(j),)) for j in range(n_sites)] 

    all_layers = []
    for _ in range(n_repetitions):
        all_layers.append(make_interaction_gates(odd_pairs,  rx_left=True))
        all_layers.append(make_interaction_gates(even_pairs, rx_left=True))
        all_layers.append(make_interaction_gates(odd_pairs,  rx_left=False))
        all_layers.append(make_interaction_gates(even_pairs, rx_left=False))

        all_layers.append(make_interaction_gates(even_pairs, rx_left=True))
        all_layers.append(make_interaction_gates(odd_pairs,  rx_left=True))
        all_layers.append(make_interaction_gates(even_pairs, rx_left=False))
        all_layers.append(make_interaction_gates(odd_pairs,  rx_left=False))

        all_layers.append(make_potential_gates())
    
    
    remapped_layers = remap_layered_gate_indices(all_layers, n_sites)
    merged_layers = merge_single_qubit_gates(remapped_layers, with_names=False, use_TN=use_TN)

    # Return for tensor network code
    if return_format == 'layered':
        return merged_layers

    # Return: flat list of np arrays
    if return_format == 'flat':
        if use_TN:
            return merged_layers #already reshaped in merge_single qubit gates
        else:
            return [gate.reshape(4, 4) for layer in merged_layers for gate, _ in layer]


