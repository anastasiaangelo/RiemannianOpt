# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Function for constructing a parameterized version of a circuit."""

from __future__ import annotations

from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, RXGate, RYGate, RZGate, UnitaryGate
from qiskit.synthesis import two_qubit_cnot_decompose
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator

import jax.numpy as jnp
from typing import Literal
from scipy.linalg import expm
import numpy as np
from typing import Tuple, List

from .from_connectivity import _allocate_parameters

I = jnp.eye(2)
X = jnp.asarray([[0., 1.],[1., 0.]])
Y = jnp.asarray([[0., -1.j],[1.j, 0.]])
Z = jnp.asarray([[1., 0.],[0.,-1.]])
XX, YY, ZZ = jnp.kron(X,X), jnp.kron(Y,Y), jnp.kron(Z,Z)
XI, IX = jnp.kron(X,I), jnp.kron(I,X)
YI, IY = jnp.kron(Y,I), jnp.kron(I,Y)
ZI, IZ = jnp.kron(Z,I), jnp.kron(I,Z)

PAULI_TO_GATE = {
    ('X', 'X'): RXXGate,
    ('Y', 'Y'): RYYGate,
    ('Z', 'Z'): RZZGate,
}

SINGLE_QUBIT_GATES = {
    'X': RXGate,
    'Y': RYGate,
    'Z': RZGate,
}

def parametrize_circuit(
    qc: QuantumCircuit,
    /,
    *,
    parameter_name: str = "theta",
) -> tuple[QuantumCircuit, list[float | None]]:
    r"""Create a parametrized version of a circuit.

    Given a quantum circuit, constructs another quantum circuit which is identical
    except that any gates with numerical parameters are replaced by gates (of the same
    type) with free parameters. The new circuit is returned along with a list containing
    the original values of the parameters.

    Args:
        qc: The quantum circuit to parametrize.
        parameter_name: Name for the :class:`~qiskit.circuit.ParameterVector`
            representing the free parameters in the returned ansatz circuit.

    Returns:
        ``(ansatz, parameter_values)`` such that ``ansatz.assign_parameters(parameter_values)``
        is identical to ``qc`` as long as ``qc`` did not already contain parameters.
        If ``qc`` already had parameters, then ``parameter_values`` will contain ``None``
        at the entries corresponding to those parameters.

    Example:
    ========

    Consider the following circuit as an example:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:
       :context: reset

       from qiskit import QuantumCircuit

       qc = QuantumCircuit(6)
       qc.rx(0.4, 0)
       qc.ryy(0.2, 2, 3)
       qc.h(2)
       qc.rz(0.1, 2)
       qc.rxx(0.3, 0, 1)
       qc.rzz(0.3, 0, 1)
       qc.cx(2, 1)
       qc.s(1)
       qc.h(4)
       qc.draw("mpl")

    If the above circuit is passed to :func:`.parametrize_circuit`, it will return an ansatz
    obtained from this circuit by replacing numerical parameters with free parameters:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:
       :context: close-figs

       from qiskit_addon_aqc_tensor import parametrize_circuit

       ansatz, initial_params = parametrize_circuit(qc)
       ansatz.draw("mpl")

    Further, the :func:`.parametrize_circuit` function provides parameters which, when bound to the ansatz, will result in a circuit identical to the original one:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:
       :context: close-figs

       ansatz.assign_parameters(initial_params).draw("mpl")

    If the original circuit already contained parameters, then the returned parameter values
    will contain ``None`` at the entries corresponding to those parameters, and the preceding
    code will not work. The following example shows how to recover the original circuit
    in this case.

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:
       :context: close-figs

       from qiskit.circuit import Parameter

       qc = QuantumCircuit(3)
       alpha1 = Parameter("alpha1")
       alpha2 = Parameter("alpha2")
       qc.ry(alpha1, [0])
       qc.rz(0.1, [0])
       qc.ry(alpha2, [1])
       qc.rz(alpha1, [1])
       qc.ry(0.2, [2])
       qc.rz(0.3, [2])
       ansatz, initial_params = parametrize_circuit(qc)
       ansatz.assign_parameters(
           {
               param: val
               for param, val in zip(ansatz.parameters, initial_params)
               if val is not None
           },
           inplace=True,
       )
       ansatz.draw("mpl")
    """
    ansatz = QuantumCircuit(*qc.qregs, *qc.cregs)
    param_vec = ParameterVector(parameter_name)
    initial_params: list[float | None] = []

    for inst in qc.data:
        operation = inst.operation
        original_params = operation.params
        fixed_indices = [
            i for i, val in enumerate(original_params) if not isinstance(val, Parameter)
        ]
        if fixed_indices:
            # Replace all non-Parameter entries with parameters
            operation = operation.copy()
            params = operation.params
            allocated_params, _ = _allocate_parameters(param_vec, len(fixed_indices))
            for i, param in zip(fixed_indices, allocated_params):
                params[i] = param
                initial_params.append(original_params[i])
        ansatz.append(operation, inst.qubits, inst.clbits)

    for i, param in enumerate(ansatz.parameters):
        if param in qc.parameters:
            initial_params.insert(i, None)

    return ansatz, initial_params


def extract_gate_matrices(qc):
    """
    Extracts unitary matrices and qubit indices from a Qiskit circuit.
    Returns a list of (matrix, [qubit indices]) tuples.
    """
    gate_seq = []
    for instr, qargs, _ in qc.data:
        try:
            U = instr.to_matrix()
        except Exception:
            U = instr.to_operator().data  # fallback
        qubit_idxs = [qc.find_bit(q).index for q in qargs]
        gate_seq.append((U, qubit_idxs))
    return gate_seq


def get_step_size(
        system: Literal['custom', 'ising-1d', 'heisenberg', 'molecular', 'fermi-hubbard-1d'],
        num_sites: int,
        **kwargs
)-> int:
    """
    Calculate the step size for the Trotterization of a quantum circuit.

    Args:
        system (str): The type of quantum system (e.g., 'standard', 'ising-1d', etc.).
        num_sites (int): Number of sites in the system.

    Returns:
        int: The step size for the Trotterization.
    """
    if system == 'ising-1d' or system == 'heisenberg':
        return int((num_sites / 2) + ((num_sites-1) // 2))
    elif system == 'fermi-hubbard-1d':
        return int(3 * num_sites - 2)
    elif system == 'molecular': # to be revised
        pauli_terms = kwargs.get('pauli_terms')
        return int(pauli_terms)
    elif system == 'custom':
        pauli_terms = kwargs.get('pauli_terms')
        if pauli_terms is None:
            raise ValueError("`pauli_terms` must be provided for system='custom'")

        # Count unique qubit pairs (i, j) with i < j
        unique_pairs = set()
        for i, j, _, _, _ in pauli_terms:
            key = tuple(sorted((i, j)))
            unique_pairs.add(key)

        return len(unique_pairs)

    else:
        raise ValueError(f"Unknown system type: {system}")


def transform_to_bw_qc_unitarygate(
    num_sites: int,
    n_repetitions: int,
    system: Literal['standard', 'ising-1d', 'heisenberg', 'molecular', 'fermi-hubbard-1d', 'pxp'],
    gates: jnp.ndarray
) -> QuantumCircuit:
    """
    Convert a list of gates into a Qiskit QuantumCircuit based on system type.
    """
    qc = QuantumCircuit(num_sites)
    g_idx = 0

    if system == 'pxp':
        # For each step: [odd3, even2, odd3, even2, even2, odd3, even2, odd3]
        odd_pattern = [True, False, True, False, False, True, False, True] * n_repetitions

        for layer_idx, is_odd in enumerate(odd_pattern):
            qubit_range = range(0, num_sites - 1, 2) if is_odd else range(1, num_sites - 1, 2)
            label = "Odd" if is_odd else "Even"
            for i in qubit_range:
                if g_idx >= len(gates): break
                gate = UnitaryGate(gates[g_idx], label=f"{label}{layer_idx}_{i}")
                qc.append(gate, [i, i + 1])
                g_idx += 1
    else:
        step = 0

        while g_idx < len(gates):
            # Odd layer
            for i in range(0, num_sites - 1, 2):
                if g_idx >= len(gates): break
                gate_qiskit = UnitaryGate(gates[g_idx], label=f'Odd{step}_{i}')
                qc.append(gate_qiskit, [i, i + 1])
                g_idx += 1

            # Even layer
            for i in range(1, num_sites - 1, 2):
                if g_idx >= len(gates): break
                gate_qiskit = UnitaryGate(gates[g_idx], label=f'Even{step}_{i}')
                qc.append(gate_qiskit, [i, i + 1])
                g_idx += 1

            step += 1

    return qc


def transform_to_bw_qc_unitarygate_names(
    n_sites: int,
    gates: jnp.ndarray,
    first_layer_odd: bool = True,
    labels: list = None  # Optional: list of gate labels
) -> QuantumCircuit:
    """
    Builds a brickwall QuantumCircuit from a list of 2-qubit gates, applying
    them in alternating odd-even or even-odd layers.

    Args:
        n_sites (int): Number of qubits.
        gates (jnp.ndarray): Stack of 4x4 gates (or list).
        first_layer_odd (bool): Whether to start with odd layer.
        labels (list): Optional labels for the gates.

    Returns:
        QuantumCircuit: The constructed Qiskit circuit.
    """
    qc = QuantumCircuit(n_sites)
    g_idx = 0
    step = 0

    while g_idx < len(gates):
        if first_layer_odd:
            # Odd layer
            for i in range(0, n_sites - 1, 2):
                if g_idx >= len(gates): break
                label = labels[g_idx] if labels else f'Odd{step}_{i}'
                gate_q = UnitaryGate(np.array(gates[g_idx]), label=label)
                qc.append(gate_q, [i, i + 1])
                g_idx += 1

            # Even layer
            for i in range(1, n_sites - 1, 2):
                if g_idx >= len(gates): break
                label = labels[g_idx] if labels else f'Even{step}_{i}'
                gate_q = UnitaryGate(np.array(gates[g_idx]), label=label)
                qc.append(gate_q, [i, i + 1])
                g_idx += 1
        else:
            # Even layer
            for i in range(1, n_sites - 1, 2):
                if g_idx >= len(gates): break
                label = labels[g_idx] if labels else f'Even{step}_{i}'
                gate_q = UnitaryGate(np.array(gates[g_idx]), label=label)
                qc.append(gate_q, [i, i + 1])
                g_idx += 1

            # Odd layer
            for i in range(0, n_sites - 1, 2):
                if g_idx >= len(gates): break
                label = labels[g_idx] if labels else f'Odd{step}_{i}'
                gate_q = UnitaryGate(np.array(gates[g_idx]), label=label)
                qc.append(gate_q, [i, i + 1])
                g_idx += 1

        step += 1

    return qc


def transform_layered_gates_to_qc_with_names(n_sites, named_layered_gates):
    qc = QuantumCircuit(n_sites)

    for layer in named_layered_gates:
        for entry in layer:
            if len(entry) == 3:
                gate, sites, name = entry
            elif len(entry) == 2:
                gate, sites = entry
                name = None
            else:
                raise ValueError("Each gate must be a (gate, sites) or (gate, sites, name) tuple.")

            if len(sites) == 1:
                qc.append(UnitaryGate(gate, label=name), [sites[0]])
            elif len(sites) == 2:
                q0, q1 = sites
                if gate.shape == (2, 2, 2, 2):
                    gate = gate.transpose(2, 3, 0, 1).reshape(4, 4)
                qc.append(UnitaryGate(gate, label=name), [q0, q1])
            else:
                raise ValueError(f"Unsupported number of qubits: {len(sites)}")

    return qc


def decompose_unitary_gate(
        qc: QuantumCircuit
)->QuantumCircuit:
    """
    Decomposes a UnitaryGate into its constituent gates and adds them to the circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to which the gates will be added.
        gate (UnitaryGate): The UnitaryGate to be decomposed.
    """
    decomposed_qc = QuantumCircuit(qc.num_qubits)

    for instr, qargs, _ in qc.data:
        if isinstance(instr, UnitaryGate) and instr.num_qubits == 2:
            # Decompose and append
            subcircuit = two_qubit_cnot_decompose(instr.to_matrix())
            decomposed_qc.compose(subcircuit, qubits=qargs, inplace=True)
        else:
            # Append any other gates directly
            decomposed_qc.append(instr, qargs)

    return decomposed_qc


def brickwall_gate_generic(i, j, t, is_edge=False, is_top=False, **kwargs):
    """
    Generate a 2-qubit gate from a generic Hamiltonian term.

    Parameters:
        i, j (int): qubit indices
        t (float): timestep
        is_edge (bool): unused for now, but included for compatibility
        is_top (bool): unused for now, but included for compatibility
        kwargs:
            - hamiltonian_terms: a dict or callable that returns a 4x4 H for (i, j)

    Returns:
        jnp.ndarray: 4x4 unitary matrix U = exp(-i t H_ij)
    """
    H_map = kwargs.get('hamiltonian_terms', None)
    if H_map is None:
        raise ValueError("Missing 'hamiltonian_terms' in kwargs.")

    # H_map can be a dict or a callable
    if callable(H_map):
        H_ij = H_map(i, j)
    else:
        H_ij = H_map.get((i, j), None)
        if H_ij is None:
            raise ValueError(f"No Hamiltonian term found for pair ({i},{j}).")

    if isinstance(H_ij, jnp.ndarray):
        H_ij = jnp.array(H_ij)

    U_ij = expm(-1j * t * H_ij)
    return jnp.asarray(U_ij)


def get_brickwall_trotter_gates_generic(
    t, 
    n_sites, 
    n_repetitions=1, 
    degree=2, 
    use_TN=False, 
    **kwargs
):
    """
    Generate Trotter gates in a brickwall pattern for a generic 2-local Hamiltonian.

    Parameters:
        t (float): total time
        n_sites (int): number of qubits/spins
        n_repetitions (int): number of Trotter repetitions
        degree (int): Trotter order (1, 2, or 4 supported)
        gate_generator (callable): function that takes (i, j, dt, is_edge, is_top, **kwargs) and returns 2-qubit unitary
        use_TN (bool): whether to return gates reshaped as tensor network format
        kwargs: extra arguments for gate_generator

    Returns:
        jnp.ndarray of shape (num_gates, 2, 2, 2, 2) if use_TN else (num_gates,)
    """
    assert degree in [1, 2, 4], "Only degrees 1, 2, and 4 are supported."
    assert n_sites >= 2 and n_sites % 2 == 0, "Even number of sites required."

    dt = t / n_repetitions
    padded_sites = n_sites if n_sites % 2 == 0 else n_sites + 1
    dummy_site = padded_sites - 1 if n_sites % 2 != 0 else None
    
    def make_layer(pairs, is_edge_layer=False, is_top_layer=False):
        gates = []
        for i, j in pairs:
            if dummy_site is not None and (i == dummy_site or j == dummy_site):
                continue  # skip dummy qubit
            g = brickwall_gate_generic(
                i, j, dt / degree,
                is_edge=(is_edge_layer and (i == 0 or j == padded_sites - 2)),
                is_top=(is_top_layer and i == 0),
                **kwargs
            )
            gates.append(g)
        return gates

        
    # Define odd and even layer site pairs
    odd_pairs = [(i, i + 1) for i in range(0, padded_sites - 1, 2)]
    even_pairs = [(i, i + 1) for i in range(1, padded_sites - 1, 2)]

    if degree in [1, 2]:
        L1 = make_layer(odd_pairs, is_edge_layer=True, is_top_layer=True)
        L2 = make_layer(even_pairs, is_edge_layer=False, is_top_layer=False)

        L1_squared = [g @ g for g in L1]
        L2_squared = [g @ g for g in L2]

        if degree == 1:
            gates = (L1 + L2) * n_repetitions
        else:
            gates = L1 + L2_squared
            for _ in range(n_repetitions - 1):
                gates += L1_squared + L2_squared
            gates += L1
    else:
        s2 = 1 / (4 - 4 ** (1 / 3))
        V1 = list(brickwall_gate_generic(2*s2*dt, n_sites, n_repetitions=2, degree=2, **kwargs))
        V2 = list(brickwall_gate_generic((1-4*s2)*dt, n_sites, n_repetitions=2, degree=2, **kwargs))
        lim = len(odd_pairs)

        V11 = [V1[j] @ V1[j] for j in range(lim)]
        V12 = [V1[j] @ V2[j] for j in range(lim)]
        V21 = [V2[j] @ V1[j] for j in range(lim)]

        repeated_gates = V1[lim:-lim] + V12 + V2[lim:-lim] + V21 + V1[lim:-lim]
        gates = V1[:lim] + repeated_gates

        for _ in range(n_repetitions - 1):
            gates += V11 + repeated_gates
        gates += V1[:lim]

    gates = jnp.asarray(gates)
    if use_TN:
        gates = gates.reshape((len(gates), 2, 2, 2, 2))

    return gates


# def truncate_ansatz(ansatz: QuantumCircuit, num_layers: int):
#     new_circuit = QuantumCircuit(ansatz.num_qubits)
#     count = 0
#     layers_added = 0
#     seen_qubits = set()

#     for instr_tuple in ansatz.data:
#         instr = instr_tuple.operation
#         qargs = instr_tuple.qubits
#         cargs = instr_tuple.clbits

#         new_circuit.append(instr, qargs, cargs)
#         count += 1

#         # Track seen qubit objects (not index)
#         for q in qargs:
#             seen_qubits.add(q)

#         if len(seen_qubits) == ansatz.num_qubits:
#             layers_added += 1
#             seen_qubits = set()

#         if layers_added == num_layers:
#             break

#     return new_circuit

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit

def truncate_ansatz(ansatz: QuantumCircuit, num_layers: int) -> QuantumCircuit:
    dag = circuit_to_dag(ansatz)
    layers = list(dag.layers())  # List of dicts with key "graph"

    # Start a new DAGCircuit from scratch
    new_dag = DAGCircuit()
    new_dag.name = f"truncated_{num_layers}_layers"
    new_dag.add_qreg(ansatz.qregs[0])
    if ansatz.cregs:
        new_dag.add_creg(ansatz.cregs[0])

    for i, layer in enumerate(layers):
        if i >= num_layers:
            break
        for node in layer['graph'].op_nodes():
            new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

    return dag_to_circuit(new_dag)


def truncate_parameters(full_ansatz, full_params, N):
    """
    Truncate the ansatz to the first N DAG layers and extract corresponding parameter values.
    """
    dag = circuit_to_dag(full_ansatz)
    layers = list(dag.layers())

    # Create a new DAGCircuit from scratch
    truncated_dag = DAGCircuit()
    truncated_dag.add_qreg(full_ansatz.qregs[0])
    if full_ansatz.cregs:
        truncated_dag.add_creg(full_ansatz.cregs[0])

    # Add first N layers of gates
    for layer in layers[:N]:
        for node in layer["graph"].op_nodes():
            truncated_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

    truncated_circuit = dag_to_circuit(truncated_dag)
    used_params = list(truncated_circuit.parameters)

    # Match to initial values
    param_indices = [list(full_ansatz.parameters).index(p) for p in used_params]
    import numpy as np
    truncated_values = np.array(full_params)[param_indices]

    return truncated_circuit, truncated_values


def truncate_qiskit_circuit(circuit: QuantumCircuit, num_layers: int) -> QuantumCircuit:
    """
    Returns a new circuit truncated after `num_layers` DAG layers.
    """
    dag = circuit_to_dag(circuit)
    layers = list(dag.layers())
    truncated_dag = dag.copy_empty_like()
    
    for i, layer in enumerate(layers):
        if i >= num_layers:
            break
        for node in layer['graph'].op_nodes():
            truncated_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
    
    return dag_to_circuit(truncated_dag)


from typing import Tuple

def remap_gate_qubits(gate: jnp.ndarray, orig_pair: Tuple[int, int], new_pair: Tuple[int, int]) -> jnp.ndarray:
    """
    Remap a 2-qubit gate acting originally on (q0, q1) to (q0', q1').
    If the relative order is flipped, swap the qubit legs.
    """
    orig_dir = orig_pair[0] < orig_pair[1]
    new_dir = new_pair[0] < new_pair[1]
    swap = orig_dir != new_dir

    if gate.shape == (4, 4):
        if not swap:
            return gate
        perm = jnp.array([0, 2, 1, 3])
        return gate[jnp.ix_(perm, perm)]
    elif gate.shape == (2, 2, 2, 2):
        if not swap:
            return gate
        return jnp.transpose(gate, (1, 0, 3, 2))
    else:
        raise ValueError("Gate must be 4x4 or (2,2,2,2)")

    

def get_qubit_mapping(
    n_sites: int,
    qubit_direction: str = 'left-to-right'     
) -> dict:
    """
    Returns a mapping from input qubit indices to your internal convention:
    - internal layout is qubit top-to-bottom (q[0] at top)
    - you specify how the input circuit is laid out
    """
    if qubit_direction in ['top-to-bottom', 'right-to-left']:
        # No remapping needed
        mapping = {i: i for i in range(n_sites)}
    elif qubit_direction in ['bottom-to-top', 'left-to-right']:
        # Reverse the order of qubits
        mapping = {i: n_sites - 1 - i for i in range(n_sites)}
    else:
        raise ValueError(f"Unsupported qubit_direction: {qubit_direction}")

    return mapping


def remap_brickwall_gates(
    gates: jnp.ndarray,
    labels: List[str],
    n_sites: int,
    qubit_direction: str = 'left-to-right'
) -> List[Tuple[str, jnp.ndarray]]:
    """
    Remap a list of 2-qubit gates and carry their labels along.
    Returns a list of (label, remapped_gate) pairs, reordered for brickwall layout.
    """
    mapping = get_qubit_mapping(n_sites, qubit_direction)

    remapped_labelled_gates = []
    g_idx = 0

    # Odd layer
    odd_layer = []
    for i in range(0, n_sites - 1, 2):
        if g_idx >= len(gates): break
        orig_pair = (i, i + 1)
        new_pair = (mapping[i], mapping[i + 1])
        remapped = remap_gate_qubits(gates[g_idx], orig_pair, new_pair)
        label = labels[g_idx]
        odd_layer.append((min(new_pair), label, remapped))
        g_idx += 1

    # Even layer
    even_layer = []
    for i in range(1, n_sites - 1, 2):
        if g_idx >= len(gates): break
        orig_pair = (i, i + 1)
        new_pair = (mapping[i], mapping[i + 1])
        remapped = remap_gate_qubits(gates[g_idx], orig_pair, new_pair)
        label = labels[g_idx]
        even_layer.append((min(new_pair), label, remapped))
        g_idx += 1

    # Combine layers, sorted by physical placement
    sorted_odd = sorted(odd_layer, key=lambda x: x[0])
    sorted_even = sorted(even_layer, key=lambda x: x[0])

    remapped_labelled_gates.extend([(label, gate) for _, label, gate in sorted_odd])
    remapped_labelled_gates.extend([(label, gate) for _, label, gate in sorted_even])

    return remapped_labelled_gates


def build_logical_left_to_right_circuit(n_sites, gates):
    qreg = QuantumRegister(n_sites, "q")
    qc = QuantumCircuit(qreg)
    wire_order = list(reversed(qreg))  # left-to-right

    g_idx = 0
    for i in range(0, n_sites - 1, 2):
        if g_idx >= len(gates): break
        qc.append(UnitaryGate(np.array(gates[g_idx]), label=f"G{g_idx}"), [wire_order[i], wire_order[i + 1]])
        g_idx += 1
    for i in range(1, n_sites - 1, 2):
        if g_idx >= len(gates): break
        qc.append(UnitaryGate(np.array(gates[g_idx]), label=f"G{g_idx}"), [wire_order[i], wire_order[i + 1]])
        g_idx += 1
    return qc

def remap_layered_gate_indices(
    layered_gates: list[list[tuple]],
    n_sites: int,
    qubit_direction: str = "left-to-right"
) -> list[list[tuple]]:
    """
    Remaps layered gate indices from logical to physical qubit layout,
    flipping gate tensors if needed. Converts all gates to complex128 and copies them.
    """
    mapping = get_qubit_mapping(n_sites, qubit_direction)
    remapped_layers = []

    for layer in layered_gates:
        remapped_layer = []
        for gate_info in layer:
            if len(gate_info) == 3:
                gate, sites, name = gate_info
            else:
                gate, sites = gate_info
                name = None

            remapped_sites = [mapping[s] for s in sites]

            # Ensure numerical consistency (always apply this)
            gate = np.array(gate, dtype=np.complex128).copy()

            if len(sites) == 2:
                orig_pair = tuple(sites)
                new_pair = tuple(remapped_sites)
                gate = remap_gate_qubits(gate, orig_pair, new_pair)

            if name is not None:
                remapped_layer.append((gate, remapped_sites, name))
            else:
                remapped_layer.append((gate, remapped_sites))

        remapped_layers.append(remapped_layer)

    return remapped_layers


def qiskit_circuit_to_layers(qc: QuantumCircuit):
    """
    Convert Qiskit circuit to a list of layers (no time slicing, all gates treated sequentially).
    Each layer is a list of (unitary, qubit_indices).
    """
    layers = []
    current_layer = []

    for instr in qc.data:
        gate, qargs, _ = instr
        qubit_indices = [qc.find_bit(q).index for q in qargs]
        matrix = np.array(Operator(gate).data)

        current_layer.append((matrix, qubit_indices))

    # For now, treat the entire circuit as one "layer"
    layers.append(current_layer)
    return layers
