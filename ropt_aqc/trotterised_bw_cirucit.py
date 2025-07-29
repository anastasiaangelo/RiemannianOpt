import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.numpy import array, cos, sin, exp, arange, asarray


from jax import config as c
c.update("jax_enable_x64", True)

I = jnp.eye(2)
X = jnp.asarray([[0., 1.],[1., 0.]])
Y = jnp.asarray([[0., -1.j],[1.j, 0.]])
Z = jnp.asarray([[1., 0.],[0.,-1.]])
XX, YY, ZZ = jnp.kron(X,X), jnp.kron(Y,Y), jnp.kron(Z,Z)
XI, IX = jnp.kron(X,I), jnp.kron(I,X)
YI, IY = jnp.kron(Y,I), jnp.kron(I,Y)
ZI, IZ = jnp.kron(Z,I), jnp.kron(I,Z)

PAULI_MAP = {
    'I': I,
    'X': X,
    'Y': Y,
    'Z': Z
}

def brickwall_gate(t, is_edge=False, is_top=False, hamiltonian='ising-1d', **kwargs):
    J = kwargs.get('J', 0.)
    g = kwargs.get('g', [0., 0.])
    h = kwargs.get('h', [0., 0.])
    
    if hamiltonian == 'ising-1d':
        g1, g2 = g[0] / 2, g[1] / 2
        h1, h2 = h[0] / 2, h[1] / 2

        if is_edge:
            if is_top:
                g1, h1 = g[0], h[0]
                g2, h2 = 0.0, 0.0
            else:
                g1, h1 = 0.0, 0.0
                g2, h2 = g[1], h[1]

        # Compose evolution as product of exponentials
        U = (
            expm(-1j * t * g1 * XI) @
            expm(-1j * t * h1 * ZI) @
            expm(-1j * t * J * ZZ) @
            expm(-1j * t * h2 * IZ) @
            expm(-1j * t * g2 * IX)
        )
        return U
    
    elif hamiltonian=='heisenberg':
        h1, h2 = h[0]/2, h[1]/2
        if is_edge: 
            if is_top: h1 = h[0]
            else: h2 = h[1] 
        op1, op2 = [XX,YY,ZZ], [[XI,IX],[YI,IY],[ZI,IZ]]
        exp = jnp.zeros_like(XX)
        for i in range(3):
            exp += (J[i]*op1[i] + h1[i]*op2[i][0] + h2[i]*op2[i][1])
        gate = expm(-1j*t*exp)
        return gate    
    
# Spin Systems

def build_make_gate(hamiltonian, J, g=None, h=None):
    "Build 2-qubit gates for spin systems, Ising 1D and Heisenberg"
    def make_gate(i, j, dt, is_edge, is_top):
        args = {
            "dt": dt,
            "is_edge": is_edge,
            "is_top": is_top,
            "hamiltonian": hamiltonian,
            "J": J[(i, j)]
        }
        if hamiltonian == "ising-1d":
            args["g"] = [g[i], g[j]]
            args["h"] = [h[i], h[j]]
        elif hamiltonian == "heisenberg":
            args["h"] = [h[i], h[j]]
        else:
            raise ValueError(f"Unsupported Hamiltonian: {hamiltonian}")
        
        return brickwall_gate(
            args["dt"],
            is_edge=args["is_edge"],
            is_top=args["is_top"],
            hamiltonian=args["hamiltonian"],
            J=args["J"],
            g=args.get("g"),
            h=args["h"]
        )

    return make_gate


def get_brickwall_trotter_gates_generic(
    t, n_sites, n_repetitions=1, degree=2, make_gate_fn=None, use_TN=False
):
    """
    General brickwall trotter gate builder for arbitrary 1D Hamiltonians.

    Parameters:
        t (float): total simulation time
        n_sites (int): number of qubits (must be >= 2)
        n_repetitions (int): number of Trotter steps
        degree (int): Trotter order (1, 2, or 4)
        make_gate_fn (callable): function (i, j, dt, is_edge, is_top) -> 4x4 gate matrix
        use_TN (bool): whether to reshape output as (N,2,2,2,2)

    Returns:
        jnp.ndarray: array of 4x4 gates (or shaped for TN if use_TN=True)
    """

    if make_gate_fn is None:
        raise ValueError("You must provide a make_gate_fn callback.")

    assert n_sites >= 2, "At least 2 sites required."
    assert degree in [1, 2, 4], "Only Trotter degrees 1, 2, or 4 supported."

    dt = t / (n_repetitions * degree) if degree in [1, 2] else t / n_repetitions

    N_odd = n_sites // 2
    N_even = n_sites // 2 - (1 if n_sites % 2 == 0 else 0)

    odd_pairs = [(i, i + 1) for i in range(0, n_sites - 1, 2)]
    even_pairs = [(i, i + 1) for i in range(1, n_sites - 1, 2)]

    if degree in [1, 2]:

        def build_layer(pairs, is_edge_layer=False):
            gates = []
            for idx, (i, j) in enumerate(pairs):
                is_edge = is_edge_layer and (idx == 0 or idx == len(pairs) - 1)
                is_top = (i == 0)
                gate = make_gate_fn(i, j, dt, is_edge, is_top)
                gates.append(gate)
            return gates

        L1 = build_layer(odd_pairs, is_edge_layer=True)
        L2 = build_layer(even_pairs, is_edge_layer=False)

        if degree == 1:
            gates = L1 + L2
            for _ in range(n_repetitions - 1):
                gates += L1 + L2
        else:
            L1_squared = [g @ g for g in L1]
            L2_squared = [g @ g for g in L2]
            gates = L1 + L2_squared
            for _ in range(n_repetitions - 1):
                gates += L1_squared + L2_squared
            gates += L1

    elif degree == 4:
        s2 = (4 - 4 ** (1 / 3)) ** -1
        gates_s1 = get_brickwall_trotter_gates_generic(
            2 * s2 * dt, n_sites, n_repetitions=2, degree=2, make_gate_fn=make_gate_fn, use_TN=False
        )
        gates_s2 = get_brickwall_trotter_gates_generic(
            (1 - 4 * s2) * dt, n_sites, n_repetitions=1, degree=2, make_gate_fn=make_gate_fn, use_TN=False
        )

        lim = len(odd_pairs)

        V11 = [g @ g for g in gates_s1[:lim]]
        V12 = [g1 @ g2 for g1, g2 in zip(gates_s1[:lim], gates_s2[:lim])]
        V21 = [g2 @ g1 for g1, g2 in zip(gates_s1[:lim], gates_s2[:lim])]

        repeated = (
            list(gates_s1[lim:-lim])
            + V12
            + list(gates_s2[lim:-lim])
            + V21
            + list(gates_s1[lim:-lim])
        )

        gates = list(gates_s1[:lim]) + repeated
        for _ in range(n_repetitions - 1):
            gates += V11 + repeated
        gates += list(gates_s1[:lim])

    gates = jnp.asarray(gates)
    if use_TN:
        gates = gates.reshape((len(gates), 2, 2, 2, 2))
    return gates


# Fermionic systems

fermionic_swap = jnp.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, -1]])


def fermionic_simulation_gate(Tpq, Vpq, t, include_swap=True): 
    if include_swap:
        return jnp.array([
            [1., 0., 0., 0.],
            [0., -1j * jnp.sin(Tpq * t), jnp.cos(Tpq * t), 0.],
            [0., jnp.cos(Tpq * t), -1j * jnp.sin(Tpq * t), 0.],
            [0., 0., 0., -jnp.exp(-1j * Vpq * t)]
        ])
    else:
        return jnp.array([
            [1., 0., 0., 0.],
            [0., jnp.cos(Tpq * t), -1j * jnp.sin(Tpq * t), 0.],
            [0., -1j * jnp.sin(Tpq * t), jnp.cos(Tpq * t), 0.],
            [0., 0., 0., jnp.exp(-1j * Vpq * t)]
        ])


def build_make_gate_fermionic(T, V):
    """
    Returns a make_gate function compatible with generic Trotter gate builders.
    Automatically expands T and V into 2-qubit-compatible dictionaries.
    
    Args:
        T: (n_orbitals, n_orbitals) hopping matrix
        V: (n_orbitals,) on-site interaction values
    
    Returns:
        make_gate(i, j, dt, is_edge, is_top) -> 4x4 gate
    """
    n_orbitals = T.shape[0]
    n_spin_orbitals = 2 * n_orbitals

    T_dict = {
        (i, j): T[i, j]
        for i in range(n_spin_orbitals)
        for j in range(n_spin_orbitals)
        if T[i // 2, j // 2] != 0
    }

    V_dict = {
        (2*p, 2*p+1): V[p]
        for p in range(n_orbitals)
    }

    def make_gate(i, j, dt, is_edge, is_top):
        Tpq = T_dict.get((i, j), T_dict.get((j, i), 0.0))
        Vpq = V_dict.get((i, j), V_dict.get((j, i), 0.0))
        return fermionic_simulation_gate(Tpq, Vpq, dt, include_swap=True)

    return make_gate


def get_swap_network_trotter_gates_fermionic(T, V, t, n_orbitals, degree=2, n_repetitions=1, use_TN=False):
    """
    Exact match to Isabel's get_swap_network_trotter_gates_fermi_hubbard_1d logic.

    T: (n_sites, n_sites) hopping matrix
    V: (n_sites,) interaction strengths
    n_orbitals: 2 * n_sites (assumes ordering: up_0, down_0, up_1, down_1, ...)
    """
    assert n_orbitals % 2 == 0
    n_sites = n_orbitals // 2
    assert T.shape == (n_sites, n_sites)
    assert len(V) == n_sites

    dt = t / n_repetitions
    if degree in [1, 2]:
        dt = dt / degree

        # kinetic gates on (1,2), (3,4), ...
        even_pairs = [(i, i+1) for i in range(1, n_orbitals - 1, 2)]
        T_coeffs = [T[i//2, (i+1)//2] for (i, _) in even_pairs]
        T_gates = [fermionic_simulation_gate(Tpq, 0.0, dt, include_swap=True) for Tpq in T_coeffs]
        T_gates_squared = [g @ g for g in T_gates]

        # interaction gates on (0,1), (2,3), ...
        V_gates = [fermionic_simulation_gate(0.0, V[p], dt, include_swap=True) for p in range(n_sites)]

        gates = []

        if degree == 1:
            one_rep = T_gates + V_gates + T_gates + [fermionic_swap for _ in V_gates]
            gates = one_rep * n_repetitions

        elif degree == 2:
            gates += T_gates + V_gates + T_gates_squared + V_gates
            for _ in range(n_repetitions - 1):
                gates += T_gates_squared + V_gates + T_gates_squared + V_gates
            gates += T_gates

        gates = jnp.asarray(gates)
        if use_TN:
            gates = gates.reshape((len(gates), 2, 2, 2, 2))
        return gates

    else:
        raise NotImplementedError("Only degree 1 and 2 supported for fermionic swap network")




def get_trotterised_gates(
    t,
    n_sites,
    n_repetitions,
    degree,
    system,
    make_gate_fn=None,
    T=None,
    V=None,
    use_TN=False,
):
    """
    Unified function to build Trotterized gates (brickwall or swap-network style),
    dispatching based on system type.

    Parameters:
    - t (float): total evolution time
    - n_sites (int): number of qubits or orbitals
    - n_repetitions (int): number of Trotter steps
    - degree (int): Trotter order (1, 2, or 4)
    - system (str): one of ['ising-1d', 'heisenberg', 'fermionic', 'custom']
    - make_gate_fn (callable): custom gate builder, only for 'custom'
    - T (array or dict): kinetic matrix (for fermionic systems)
    - V (array or dict): interaction matrix (for fermionic systems)
    - use_TN (bool): whether to reshape output for tensor network (N,2,2,2,2)

    Returns:
    - jnp.ndarray: gate sequence
    """

    if system in ["ising-1d", "heisenberg"]:
        assert make_gate_fn is not None, "Provide make_gate_fn for spin systems."
        return get_brickwall_trotter_gates_generic(
            t=t,
            n_sites=n_sites,
            n_repetitions=n_repetitions,
            degree=degree,
            make_gate_fn=make_gate_fn,
            use_TN=use_TN
        )

    elif system == "fermionic":
        assert T is not None and V is not None, "Need T, V for fermionic system."
        return get_swap_network_trotter_gates_fermionic(
            T=T,
            V=V,
            t=t,
            n_orbitals=n_sites,
            degree=degree,
            n_repetitions=n_repetitions,
            use_TN=use_TN
        )

    elif system == "custom":
        assert make_gate_fn is not None, "Provide make_gate_fn for custom Hamiltonians."
        return get_brickwall_trotter_gates_generic(
            t=t,
            n_sites=n_sites,
            n_repetitions=n_repetitions,
            degree=degree,
            make_gate_fn=make_gate_fn,
            use_TN=use_TN
        )

    else:
        raise ValueError(f"Unsupported system type: {system}")
    




def build_make_gate_from_paulis(hamiltonian_terms, n_sites):
    "Build 2-qubit gates for arbitrary Pauli strings, for QUBO sysetms for example"
    def make_gate(i, j, dt, is_edge, is_top):
        # Get terms for pair (i,j)
        key = tuple(sorted((i, j)))
        terms = hamiltonian_terms.get(key, [])

        H_ij = jnp.zeros((4, 4), dtype=jnp.complex128)  # 2-qubit Hamiltonian

        for coeff, pauli_dict in terms:
            op_i = pauli_dict.get(i, 'I')
            op_j = pauli_dict.get(j, 'I')

            Pi = PAULI_MAP[op_i]
            Pj = PAULI_MAP[op_j]
            term = coeff * jnp.kron(Pi, Pj)
            H_ij += term

        U_ij = expm(-1j * dt * H_ij)
        return U_ij

    return make_gate
