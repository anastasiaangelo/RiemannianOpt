from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RZZGate, RXXGate, RYYGate
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.quantum_info import SparseObservable

import jax.numpy as jnp
from jax.numpy import eye, kron, asarray, zeros, cos, sin, array, exp
from typing import Literal

# Pauli matrices
I = eye(2)
X = asarray([[0., 1.], [1., 0.]])
Y = asarray([[0., -1.j], [1.j, 0.]])
Z = asarray([[1., 0.], [0., -1.]])

def tensor_product(operators):
    from functools import reduce
    return reduce(kron, operators)

def pauli_string(term, num_qubits):
    """
    Builds a tensor product operator for a specific qubit pair.
    """
    ops = [I] * num_qubits
    for qubit, op in term.items():
        ops[qubit] = op

    return tensor_product(ops)


def to_diag_matrix(x, size):
        if isinstance(x, (float, int)):
            return jnp.diag(jnp.array([x] * size))
        elif isinstance(x, (list, jnp.ndarray)):
            x = jnp.array(x)
            if x.ndim == 1:
                return jnp.diag(x)
            elif x.ndim == 2:
                return x
        raise ValueError("Invalid format for parameter, must be scalar, 1D or 2D array.")


def to_jax_array(x, shape=None):
    if isinstance(x, (int, float)):
        return jnp.full(shape, x)
    else:
        x = jnp.array(x)
        if shape is not None and x.shape != shape:
            raise ValueError(f"Expected shape {shape}, got {x.shape}")
        return x


def get_hamiltonian_terms(
    num_qubits: int,
    system: Literal['custom', 'ising-1d', 'heisenberg', 'molecular', 'fermi-hubbard-1d'],
    **kwargs
):
    terms = []
    params = {}

    if system == 'custom':
        pauli_terms = kwargs.get('pauli_terms')
        assert pauli_terms is not None, "`pauli_terms` must be provided for system='custom'"

        for i, j, P1, P2, coeff in pauli_terms:
            terms.append((coeff, {i: P1, j: P2}))

        params = {'hamiltonian_terms': pauli_terms}

    elif system == 'ising-1d':
        J_raw = kwargs.get('J', [1.0]*(num_qubits-1))
        h_raw = kwargs.get('h', 1.0)
        g_raw = kwargs.get('g', 1.0)

        if isinstance(J_raw, (int, float)):
            J = jnp.eye(num_qubits, k=1) * J_raw
        else:
            J = jnp.array(J_raw)
            if J.shape != (num_qubits, num_qubits):
                raise ValueError(f"Expected J of shape {(num_qubits, num_qubits)}, got {J.shape}")

        h = to_jax_array(h_raw, (num_qubits,))
        g = to_jax_array(g_raw, (num_qubits,))
        params = {'J': J, 'h': h, 'g': g}

        for i in range(num_qubits - 1):
            if J[i, i + 1] != 0.0:
                terms.append((J[i, i + 1], {i: 'Z', i + 1: 'Z'}))

        for i in range(num_qubits):
            if h[i] != 0.0:
                terms.append((h[i], {i: 'Z'}))
            if g[i] != 0.0:
                terms.append((g[i], {i: 'X'}))

    elif system == 'heisenberg':
        J_raw = kwargs.get('J', [1.0, 1.0, -0.5])
        h_raw = kwargs.get('h', [0.75, 0.0, 0.0])
        
        J_raw = jnp.array(J_raw, dtype=jnp.float32)

        if J_raw.shape != (3,):
            raise ValueError(f"Expected J to be a 3-vector (Jx, Jy, Jz), got shape {J_raw.shape}")
        
        J_tensor = jnp.zeros((num_qubits, num_qubits, 3), dtype=jnp.float32)

        for i in range(num_qubits - 1):
            J_tensor = J_tensor.at[i, i + 1].set(J_raw)

        if isinstance(h_raw, (float, int)):
            h_raw = [h_raw, 0.0, 0.0]

        h_tensor = jnp.zeros((num_qubits, 3))
        for i in range(num_qubits):
            h_tensor = h_tensor.at[i, :].set(jnp.array(h_raw))

        for i in range(num_qubits - 1):
            if J_raw[0] != 0:
                terms.append((J_raw[0], {i: 'X', i + 1: 'X'}))
            if J_raw[1] != 0:
                terms.append((J_raw[1], {i: 'Y', i + 1: 'Y'}))
            if J_raw[2] != 0:
                terms.append((J_raw[2], {i: 'Z', i + 1: 'Z'}))

        for i in range(num_qubits):
            if h_tensor[i, 0] != 0:
                terms.append((h_tensor[i, 0], {i: 'X'}))
            if h_tensor[i, 1] != 0:
                terms.append((h_tensor[i, 1], {i: 'Y'}))
            if h_tensor[i, 2] != 0:
                terms.append((h_tensor[i, 2], {i: 'Z'}))

        params = {'J': J_tensor, 'h': h_tensor}

    else:
        raise ValueError(f"Unknown system type: {system}")

    return terms, params


def build_matrix_from_terms(terms, num_qubits):
    H = jnp.zeros((2**num_qubits, 2**num_qubits), dtype=jnp.complex128)

    for coeff, paulis in terms:
        ops = [I] * num_qubits
        for i, pauli in paulis.items():
            if pauli == 'X':
                ops[i] = X
            elif pauli == 'Y':
                ops[i] = Y
            elif pauli == 'Z':
                ops[i] = Z
        H += coeff * tensor_product(ops)
    return H


def hamiltonian_to_sparse_pauli_op(terms, num_qubits):
    pauli_ops = []
    coeffs = []

    for coeff, paulis in terms:
        pauli_str = ['I'] * num_qubits
        for qubit_index, pauli in paulis.items():
            pauli_str[qubit_index] = pauli

        # Join the list of Pauli operators into a string
        pauli_str = ''.join(pauli_str)
        
        # Create the Pauli operator and add it to the list
        pauli_ops.append(Pauli(pauli_str))
        coeffs.append(coeff)

    # Return the SparsePauliOp which is the correct format for time evolution
    return SparsePauliOp(pauli_ops, coeffs)

def hamiltonian_to_sparse_observable(terms, num_qubits):
    pauli_dict = {}
    
    for coeff, paulis in terms:
        pauli_str_list = ['I'] * num_qubits
        for qubit_index, pauli in paulis.items():
            pauli_str_list[qubit_index] = pauli
        pauli_str = ''.join(pauli_str_list)
        
        if pauli_str in pauli_dict:
            pauli_dict[pauli_str] += coeff
        else:
            pauli_dict[pauli_str] = coeff
    
    # Convert dictionary to a list of tuples: [(pauli_str, coeff), ...]
    obs_list = list(pauli_dict.items())
    return SparseObservable(obs_list)


def extract_two_qubit_hamiltonians(sparse_op: SparsePauliOp):
    """
    Extract 2-qubit Hamiltonians from a global SparsePauliOp and return a dict (i, j) -> 4x4 matrix.
    """
    h_terms = dict()

    for pauli, coeff in zip(sparse_op.paulis, sparse_op.coeffs):
        label = str(pauli)  # string like 'IXIZI...'

        # Find which qubits are active
        active = [i for i, p in enumerate(label) if p != 'I']
        if len(active) != 2:
            continue

        i, j = sorted(active)
        local_label = label[i] + label[j]  # e.g., 'XY'

        # Convert this 2-qubit Pauli string to a matrix
        p_local = Pauli(local_label)
        local_matrix = coeff * p_local.to_matrix()

        if (i, j) not in h_terms:
            h_terms[(i, j)] = local_matrix
        else:
            h_terms[(i, j)] += local_matrix

    return h_terms



def build_brickwall_circuit(
    num_qubits: int,
    depth: int,
    gate_fn,            # function to generate 2-qubit gate (e.g., RZZGate(theta))
    parameter_prefix="θ"
) -> QuantumCircuit:
    """
    Builds a brickwall circuit with arbitrary 2-qubit gates.

    Args:
        num_qubits: Number of qubits in the circuit.
        depth: Number of brickwall layers.
        gate_fn: Function that returns a Qiskit 2-qubit gate given a parameter.
        parameter_prefix: Prefix for symbolic parameters.

    Returns:
        A parameterized QuantumCircuit with a brickwall layout.
    """
    qc = QuantumCircuit(num_qubits)
    param_count = 0
    for layer in range(depth):
        offset = layer % 2  # Even layers: (0,1), (2,3)... | Odd: (1,2), (3,4)...
        for i in range(offset, num_qubits - 1, 2):
            theta = Parameter(f"{parameter_prefix}_{param_count}")
            qc.append(gate_fn(theta), [i, i+1])
            param_count += 1
    return qc


def hopping_gate(Tpq, t=1.0):
    """
    Two-qubit gaterepresenting the time evolution under the kinetic hopping term for the FH1D hamiltonian
    """
    FSG = array([
        [1., 0., 0., 0.],
        [0., cos(-Tpq*t), -1j*sin(-Tpq*t), 0.],
        [0., -1j*sin(-Tpq*t), cos(-Tpq*t), 0.],
        [0., 0., 0., 1]
        ])
    return FSG

def interaction_gate(Vpq, t):
    """
    Two-qubit gate representing the time evolution under the interaction on-site term for the FH1D hamiltonian
    """
    FSG = array([
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., -exp(-1j*Vpq*t)]
        ])
    return FSG