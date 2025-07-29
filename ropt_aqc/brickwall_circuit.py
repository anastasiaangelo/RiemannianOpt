## From Isabel's code

from functools import reduce

from jax.numpy import eye, kron, asarray
from jax import config as c
import jax.numpy as jnp
c.update("jax_enable_x64", True)

from .fermionic_systems import get_swap_network_trotter_gates_fermi_hubbard_1d, get_swap_network_trotter_gates_molecular
from .spin_systems import get_brickwall_trotter_gates_spin_chain, get_brickwall_trotter_gates_custom
from .util import get_identity_layers, apply_gate_to_mpo

from ropt_aqc.pxp_model import get_brickwall_trotter_gates_spin_chain_pxp, get_brickwall_twoqubit_layered_gates

# Define some operators
I = eye(2)
X = asarray([[0., 1.],[1., 0.]])
Y = asarray([[0., -1.j],[1.j, 0.]])
Z = asarray([[1., 0.],[0.,-1.]])
XX = kron(X,X)
YY = kron(Y,Y)
ZZ = kron(Z,Z)
XI = kron(X,I)
IX = kron(I,X)
ZI = kron(Z,I)
IZ = kron(I,Z)

def tensor_product(operators):
    return reduce(kron, operators)

def get_nlayers(degree, n_repetitions, n_sites=None, n_id_layers=0, hamiltonian='fermi-hubbard-1d'):
    if hamiltonian=='fermi-hubbard-1d':
        if degree==1:
            n_SN_layers = 4*n_repetitions
        elif degree==2:
            n_SN_layers = 3*degree*n_repetitions  # Number of layers in pure swap network
            n_SN_layers -= (degree*n_repetitions-1)  # If we absorb 
        elif degree==4:
            n_SN_layers = 20*n_repetitions+1
    elif hamiltonian=='molecular':
        assert(type(n_sites) is not type(None))
        if degree in [1,2]:
            n_SN_layers = 2*n_sites*n_repetitions  # Number of layers in pure swap network
            n_SN_layers -= (2*n_repetitions-1)  # If we absorb 
        elif degree==4:
            n_SN_layers = 10*(n_sites-1)*n_repetitions+1
    elif hamiltonian in ['ising-1d','heisenberg']:
        if degree==1:
            n_SN_layers = 2*n_repetitions
        elif degree==2: 
            n_SN_layers = 2*n_repetitions+1
        elif degree==4:
            n_SN_layers = 10*n_repetitions+1
    elif hamiltonian=='pxp':
        if degree==1:
            n_SN_layers = 8*n_repetitions
    # elif hamiltonian=='custom':
    #     assert n_sites is not None, "`n_sites` must be provided for custom Hamiltonian"
    #     if degree==1:
    #         n_SN_layers = 2*n_repetitions   # Odd + Even pair layers repeated
    #     elif degree==2: 
    #         n_SN_layers = 3*n_repetitions+1  # Odd, Even², repeated
    #     elif degree==4:
    #         n_SN_layers = 10*n_repetitions+1  # Arbitrtary heuristic
  
    else:
        raise ValueError(f"Unknown Hamiltonian type: {hamiltonian}")

    return n_SN_layers+n_id_layers



def get_initial_gates(n_sites, t, n_repetitions=1, degree=2, 
                      hamiltonian='fermi-hubbard-1d', n_id_layers=0, use_TN=True, **kwargs):
    if hamiltonian == 'fermi-hubbard-1d':
        T, V = kwargs['T'], kwargs['V']
        Vlist_start = get_swap_network_trotter_gates_fermi_hubbard_1d(T, V, t, n_sites, n_repetitions, degree, use_TN)
        first_layer_odd=False
    elif hamiltonian=='molecular':
        T, V = kwargs['T'], kwargs['V']
        Vlist_start = get_swap_network_trotter_gates_molecular(T, V, t, n_sites, degree, n_repetitions, use_TN=use_TN)
        assert(n_id_layers==0)
    elif hamiltonian in ['ising-1d', 'heisenberg']:
        Vlist_start = get_brickwall_trotter_gates_spin_chain(t, n_sites, n_repetitions, degree, hamiltonian, use_TN, **kwargs)
        if degree==1: first_layer_odd=True
        elif degree in [2,4]: first_layer_odd=False
    elif hamiltonian=='custom':
        Vlist_start = get_brickwall_trotter_gates_custom(t=t, n_sites=n_sites, n_repetitions=n_repetitions, degree=degree, use_TN=use_TN, hamiltonian_terms=kwargs['hamiltonian_terms'])
        if degree==1: first_layer_odd=True
        elif degree in [2,4]: first_layer_odd=False
        
    elif hamiltonian=='pxp':
        # Vlist_start = get_brickwall_trotter_gates_spin_chain_pxp(t=t, n_sites=n_sites, n_repetitions=n_repetitions, use_TN=use_TN, return_format='flat')
        Vlist_start = get_brickwall_twoqubit_layered_gates(t=t, n_sites=n_sites, n_repetitions=n_repetitions, use_TN=use_TN)

        first_layer_odd=True

        
    if n_id_layers>0:
        Vlist_start = list(Vlist_start)+list(get_identity_layers(n_sites, n_id_layers, first_layer_odd, use_TN))
    if hamiltonian=='pxp':
        return Vlist_start  # PXP gates are already in the correct format
    else:
        return asarray(Vlist_start)


def get_gates_per_layer(Vlist, n_sites, degree=None, n_repetitions=None,
                        n_layers=None, n_id_layers=0, hamiltonian='fermi-hubbard-1d'):
    # Ensure warning tracker is global across all calls
    if not hasattr(get_gates_per_layer, "_warned_n_set"):
        get_gates_per_layer._warned_n_set = set()

    if hamiltonian == 'pxp':
        if not hasattr(get_gates_per_layer, "_warned_n_set"):
            get_gates_per_layer._warned_n_set = set()

        # Define the fixed pattern of layer types (True = odd, False = even)
        odd_pattern = [True, False, True, False, False, True, False, True] * n_repetitions
        L = n_layers if n_layers is not None else len(odd_pattern)

        gates_per_layer, gate_sites_per_layer, layer_is_odd = [], [], []
        ptr = 0
        total_gates = len(Vlist)

        for i in range(L):
            odd = odd_pattern[i % len(odd_pattern)]
            layer_is_odd.append(odd)

            if odd:
                sites = [(j, j+1) for j in range(0, n_sites-1, 2)]
            else:
                sites = [(j, j+1) for j in range(1, n_sites-1, 2)]

            n_gates = len(sites)
            if ptr + n_gates > total_gates:
                break

            gates_per_layer.append(Vlist[ptr:ptr + n_gates])
            gate_sites_per_layer.append(sites)
            ptr += n_gates
        
        return gates_per_layer, gate_sites_per_layer, layer_is_odd

    # Standard models (Ising, Heisenberg, etc.)
    N_odd_gates = n_sites // 2
    N_even_gates = N_odd_gates - 1 if n_sites % 2 == 0 else N_odd_gates

    if n_layers is None:
        assert degree is not None and n_repetitions is not None
        n_SN_layers = get_nlayers(degree, n_repetitions, n_sites, n_id_layers, hamiltonian)
    else:
        n_SN_layers = n_layers

    if hamiltonian == 'fermi-hubbard-1d':
        odd = False
    elif hamiltonian in ['molecular', 'ising-1d', 'heisenberg']:
        odd = True
    else:
        raise ValueError(f"Unsupported hamiltonian: {hamiltonian}")

    gates_per_layer, gate_sites_per_layer, layer_is_odd = [], [], []
    lim1 = 0

    for _ in range(n_SN_layers):
        n_gates = N_odd_gates if odd else N_even_gates
        lim2 = lim1 + n_gates
        if lim2 > len(Vlist):
            break
        gates_per_layer.append(Vlist[lim1:lim2])
        layer_is_odd.append(odd)
        if odd:
            sites = [(j, j + 1) for j in range(0, n_sites - 1, 2)]
        else:
            sites = [(j, j + 1) for j in range(1, n_sites - 1, 2)]
        
        gate_sites_per_layer.append(sites[:n_gates])

        lim1 = lim2
        odd = not odd

    # Return None for gate_sites for full backward compatibility
    return gates_per_layer, gate_sites_per_layer, layer_is_odd
    

def get_layers_from_gate_sequence(Vlist):
    return [Vlist], [False]  # single "layer", non-alternating

def contract_gate_list_with_mpo(U0, Vlist, max_bond=None):
    U = U0.copy()
    for gate in Vlist:
        U = apply_gate_to_mpo(U, gate, contract=True, max_bond=max_bond)
    return U


def get_gate_qubit_pairs(Vlist, n_sites, degree, n_repetitions, n_id_layers=0, hamiltonian='ising-1d'):
    """
    Converts a flat list of gates (Vlist) into a list of layers, 
    each layer containing tuples of (q0, q1, gate), according to the swap network layout.
    """
    def get_nlayers(degree, n_repetitions, n_sites, n_id_layers, hamiltonian):
        if hamiltonian in ['ising-1d', 'heisenberg']:
            return n_repetitions * degree + n_id_layers
        else:
            raise ValueError(f"Hamiltonian {hamiltonian} not supported for layer counting.")

    N_odd_gates = n_sites // 2
    N_even_gates = N_odd_gates - (1 if n_sites % 2 == 0 else 0)

    n_layers = get_nlayers(degree, n_repetitions, n_sites, n_id_layers, hamiltonian)

    if hamiltonian in ['ising-1d', 'heisenberg', 'molecular']:
        odd = True  # Start with odd
    elif hamiltonian == 'fermi-hubbard-1d':
        odd = False  # Start with even
    else:
        raise ValueError(f"Unsupported Hamiltonian {hamiltonian}")

    gates_per_layer = []
    layer_is_odd = []
    gate_idx = 0

    for layer_idx in range(n_layers):
        layer = []
        qubit_pairs = (
            [(i, i + 1) for i in range(1, n_sites - 1, 2)] if odd else
            [(i, i + 1) for i in range(0, n_sites - 1, 2)]
        )
        for (q0, q1) in qubit_pairs:
            if gate_idx >= len(Vlist):
                print(f"⚠️ Truncating at layer {layer_idx}, only {len(Vlist)} gates available (requested {n_layers})")
                return gates_per_layer, layer_is_odd
            gate = Vlist[gate_idx]
            layer.append((q0, q1, gate))
            gate_idx += 1
        gates_per_layer.append(layer)
        layer_is_odd.append(odd)
        odd = not odd

    return gates_per_layer, layer_is_odd

