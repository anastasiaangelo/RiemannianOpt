from scipy.linalg import expm
from ropt_aqc.initialise_hamiltonians import build_matrix_from_terms
from qiskit.quantum_info import Statevector, state_fidelity , SparsePauliOp
from qiskit_algorithms import TrotterQRTE, TimeEvolutionProblem
from qiskit.primitives import Estimator


def get_qubits_from_pauli(pauli_string):
        """
        Given a Pauli string (e.g., 'IZIXII'), return a list of qubit indices 
        where the operator is not 'I'.
        """
        return [i for i, char in enumerate(pauli_string) if char != 'I']



def trotterisation(degree, H_sparse, evolution_time, n_repetitions, initial_state, num_sites, terms):
    # Second order Trotterisation
    # split terms to odd and even pairs, hamiltonian in commuting terms

    H = build_matrix_from_terms(terms, num_sites)

    # Exact evolution operator
    U_exact = expm(-1j * evolution_time * H)

    # Initial statevector
    initial_sv = Statevector(initial_state)

    # Apply exact evolution
    reference_state = initial_sv.evolve(U_exact)


    if degree == 2:
        H_A_terms = []
        H_B_terms = []

        for pauli, coeff in zip(H_sparse.paulis, H_sparse.coeffs):
            qubits_involved = get_qubits_from_pauli(pauli.to_label())
            
            if len(qubits_involved) == 2:
                q_min = min(qubits_involved)
                if q_min % 2 == 0:
                    H_A_terms.append((pauli, coeff))
                else:
                    H_B_terms.append((pauli, coeff))
            elif len(qubits_involved) == 1:
                # For single-qubit terms, assign to H_B (or split differently if preferred)
                H_B_terms.append((pauli, coeff))
            else:
                raise ValueError("Unexpected term structure.")
            

        H_A = SparsePauliOp.from_list([(p.to_label(), c) for p, c in H_A_terms])
        H_B = SparsePauliOp.from_list([(p.to_label(), c) for p, c in H_B_terms])

        from qiskit.circuit.library import PauliEvolutionGate

        dt = evolution_time / n_repetitions
        trotter_circuit = initial_state.copy()

        for _ in range(n_repetitions):
            trotter_circuit.append(PauliEvolutionGate(H_A, time=dt/2), trotter_circuit.qubits)
            trotter_circuit.append(PauliEvolutionGate(H_B, time=dt), trotter_circuit.qubits)
            trotter_circuit.append(PauliEvolutionGate(H_A, time=dt/2), trotter_circuit.qubits)

        # Now you have a second-order Trotter circuit
        final_circuit = trotter_circuit.decompose()
    
    if degree == 1:
        problem = TimeEvolutionProblem(H_sparse, initial_state=initial_state, time=evolution_time)

        trotter = TrotterQRTE(num_timesteps=n_repetitions, estimator=Estimator())
        result = trotter.evolve(problem)

        statevector = Statevector(result.evolved_state)
    
        final_circuit = result.evolved_state.decompose(reps=2).decompose("disentangler_dg").decompose(
                "multiplex1_reverse_dg"
            )

    final_state = Statevector(final_circuit)
    fidelity = state_fidelity(final_state, reference_state)

    return final_circuit, fidelity
