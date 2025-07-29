import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from typing import Tuple, Sequence

from .brickwall_circuit import get_gates_per_layer, get_layers_from_gate_sequence, contract_gate_list_with_mpo
from .tn_helpers import (get_id_mpo, canonicalize_local_tensor, merge_two_mpos_and_gate, 
                         split_tensor_into_half_canonical_mpo_pair, right_to_left_RQ_sweep,
                         merge_mpo_and_layer, compress_mpo, fully_contract_mpo, convert_mpo_to_mps, get_left_canonical_mps, inner_product_mps)
from .util import project_unitary_tangent_vectorized
from qiskit_addon_aqc_tensor.simulation import compute_overlap


def get_mpo_pairs(mpo, odd):
    """
    Get the MPO pairs for merging with a layer if a layer is 
    odd=True or even (odd=False).
    """
    if not odd: mpo_pairs = [(mpo[0],)]; start=1
    else: mpo_pairs = []; start=0
    for impo in range(start, len(mpo)-1, 2):
        mpo_pairs.append((mpo[impo], mpo[impo+1]))
    if not odd and len(mpo)%2==0: mpo_pairs.append((mpo[-1],))
    elif odd and len(mpo)%2==1: mpo_pairs.append((mpo[-1],))
    return mpo_pairs


def merge_and_truncate_mpo_and_layer_right_to_left(
    mpo_init, gates_in_layer, gate_sites=None, odd_layer=True, 
    max_bondim=128, layer_is_below=True
):
    '''
    Right-to-left contraction of an MPO layer with gate layer.
    layer_is_below corresponds to layer_is_left
    '''
    n_spin_orbitals = len(mpo_init)

    if not odd_layer:
        Q, R = canonicalize_local_tensor(mpo_init[-1], left=False)
        # print("R", R.shape)
        mpo_res = [Q]
        i_mpo_init = n_spin_orbitals - 2
    else:
        mpo_res = []
        i_mpo_init = n_spin_orbitals - 1
        R = None

    while i_mpo_init - 1 >= 0:
        site_pair = (i_mpo_init - 1, i_mpo_init)
        mpo1 = mpo_init[i_mpo_init - 1]
        mpo2 = mpo_init[i_mpo_init]
        mpo2_R = mpo2 if R is None else jnp.einsum('iabj,jk->iabk', mpo2, R)

        if gate_sites is not None and site_pair not in gate_sites:
            mpo_res.append(mpo2_R)
            if i_mpo_init - 1 == 0:
                mpo_res.append(mpo1)
                R = None
            else:
                Q, R = canonicalize_local_tensor(mpo1, left=False)
                # print("R", R.shape)
                mpo_res.append(Q)
            i_mpo_init -= 2
            continue

        gate_idx = gate_sites.index(site_pair) if gate_sites else int(i_mpo_init / 2)
        gate = gates_in_layer[gate_idx]

        merged_T = merge_two_mpos_and_gate(gate, mpo1, mpo2_R, gate_is_left=layer_is_below)
        T1, T2 = split_tensor_into_half_canonical_mpo_pair(merged_T, canonical_mode='right', max_bondim=max_bondim)

        if i_mpo_init - 1 == 0:
            mpo_res += [T2, T1]
            R = None
        else:
            Q, R = canonicalize_local_tensor(T1, left=False)
            # print("R", R.shape)
            mpo_res += [T2, Q]

        i_mpo_init -= 2

    if i_mpo_init == 0:
        mpo0 = jnp.einsum('iabj,jk->iabk', mpo_init[0], R) if R is not None else mpo_init[0]
        mpo_res.append(mpo0)

    return mpo_res[::-1]


def merge_and_truncate_mpo_and_layer_left_to_right(
    mpo_init, gates_in_layer, gate_sites=None, odd_layer=False, 
    max_bondim=128, layer_is_below=True
):
    n_spin_orbitals = len(mpo_init)

    if not odd_layer:
        Q, R = canonicalize_local_tensor(mpo_init[0], left=True)
        # print("R", R.shape)
        mpo_res = [Q]
        i_mpo_init = 1
    else:
        mpo_res = []
        i_mpo_init = 0
        R = None

    while i_mpo_init + 1 < n_spin_orbitals:
        site_pair = (i_mpo_init, i_mpo_init + 1)
        mpo1, mpo2 = mpo_init[i_mpo_init:i_mpo_init + 2]
        mpo1_R = mpo1 if R is None else jnp.einsum('ij,jabk->iabk', R, mpo1)

        if gate_sites is not None and site_pair not in gate_sites:
            mpo_res.append(mpo1_R)
            if i_mpo_init + 1 == n_spin_orbitals - 1:
                mpo_res.append(mpo2)
                R = None
            else:
                Q, R = canonicalize_local_tensor(mpo2, left=True)
                # print("R", R.shape)
                mpo_res.append(Q)
            i_mpo_init += 2
            continue

        gate_idx = gate_sites.index(site_pair) if gate_sites else int(i_mpo_init / 2)
        gate = gates_in_layer[gate_idx]

        merged_T = merge_two_mpos_and_gate(gate, mpo1_R, mpo2, gate_is_left=layer_is_below)
        T1, T2 = split_tensor_into_half_canonical_mpo_pair(merged_T, canonical_mode='left', max_bondim=max_bondim)

        if i_mpo_init + 1 == n_spin_orbitals - 1:
            mpo_res += [T1, T2]
            R = None
        else:
            Q, R = canonicalize_local_tensor(T2, left=True)
            # print("R", R.shape)
            mpo_res += [T1, Q]

        i_mpo_init += 2

    if i_mpo_init == n_spin_orbitals - 1:
        mpo_R = jnp.einsum('ij,jabk->iabk', R, mpo_init[-1]) if R is not None else mpo_init[-1]
        mpo_res.append(mpo_R)

    return mpo_res


def contract_layers_of_swap_network_with_mpo(mpo_init, gates_per_layer, layer_is_odd,  gate_sites_per_layer=None, 
                                             layer_is_left=True, max_bondim=128,
                                             get_norm=False):
    '''
    Method to merge and compress the swap network with the reference MPO
    layerwise.
    '''
    nlayers = len(gates_per_layer)
    mpo = mpo_init.copy()
    if layer_is_left: iterator=reversed(range(nlayers)) 
    else: iterator=range(nlayers)

    # Bring initial MPO into right canonical form
    mpo = right_to_left_RQ_sweep(mpo, get_norm=False)
    merge_left_to_right = True

    for layer in iterator:
        # Get the gates in the layer
        odd = layer_is_odd[layer]
        gates = gates_per_layer[layer]
        gate_sites = gate_sites_per_layer[layer] if gate_sites_per_layer is not None else None
        # Merge layer with MPO
        if merge_left_to_right: mpo = merge_and_truncate_mpo_and_layer_left_to_right(mpo, gates, gate_sites, odd, max_bondim, layer_is_left)
        else: mpo = merge_and_truncate_mpo_and_layer_right_to_left(mpo, gates, gate_sites, odd, max_bondim, layer_is_left)
        merge_left_to_right = not merge_left_to_right

    # Bring MPO into right canonical form and obtain its norm (unitarity check)
    if get_norm:
        mpo, nrm = right_to_left_RQ_sweep(mpo, get_norm=True)
        return mpo, nrm

    else: 
        return mpo


def contract_layers_of_swap_network(
        mpo_init, gates_per_layer, layer_is_odd, 
        layer_is_left=True, max_bondim=128):
    '''
    Yu's approach: method='QR-SVD'
    Gray's approach: method='QR-RQ-SVD'

    gates_per_layer: list of list per layer with Vlist_TN per layer

    This function takes an initial MPO and contracts it with
    the list of layers given. This is used to obtain the fragmented reference.

    The gates are splitted into 1-qubit tensors in this function.
    '''
    
    nlayers = len(gates_per_layer)
    iterator=reversed(range(nlayers)) if layer_is_left else range(nlayers)

    for layer in iterator:      
        odd = layer_is_odd[layer]
        gates = gates_per_layer[layer]

        # Merge layer with MPO
        mpo_res = merge_mpo_and_layer(gates, mpo_init, odd, layer_is_left=True)
        
        # Truncate the resulting MPO
        mpo_res = compress_mpo(mpo_res, max_bondim)
        mpo_init = mpo_res.copy()

    return mpo_res


def compute_cost_from_circuit_mpo(
    reference_mpo, gates_per_layer, layer_is_odd, max_bondim, gate_sites_per_layer=None
):
    circuit_mpo = get_id_mpo(len(reference_mpo))
    merge_right_to_left = True

    for l in range(len(gates_per_layer)):
        gates = gates_per_layer[l]
        odd = layer_is_odd[l]
        gate_sites = gate_sites_per_layer[l] if gate_sites_per_layer is not None else None

        if merge_right_to_left:
            circuit_mpo = merge_and_truncate_mpo_and_layer_right_to_left(
                circuit_mpo, gates, odd_layer=odd, max_bondim=max_bondim,
                **({"gate_sites": gate_sites} if gate_sites is not None else {}),
                layer_is_below=False
            )
        else:
            circuit_mpo = merge_and_truncate_mpo_and_layer_left_to_right(
                circuit_mpo, gates, odd_layer=odd, max_bondim=max_bondim,
                **({"gate_sites": gate_sites} if gate_sites is not None else {}),
                layer_is_below=False
            )

        merge_right_to_left = not merge_right_to_left

    # Frobenius norm squared: ||A - B||^2 = ||A||^2 + ||B||^2 - 2 Re Tr(A† B)
    norm_ref_sq = jnp.sum(jnp.conj(reference_mpo) * reference_mpo)
    norm_circ_sq = jnp.sum(jnp.conj(circuit_mpo) * circuit_mpo)
    overlap = jnp.sum(jnp.conj(reference_mpo) * circuit_mpo)

    cost = norm_ref_sq + norm_circ_sq - 2 * jnp.real(overlap)
    return cost


def check_full_gradient(U_mpo, Vlist_start, max_bondim, n_sites, degree, n_repetitions, n_id_layers, hamiltonian, reference_is_normalized):
    gates_per_layer, gate_sites_per_layer, layer_is_odd = get_gates_per_layer(
        Vlist_start, n_sites, degree, n_repetitions, n_id_layers=n_id_layers, hamiltonian=hamiltonian
    )

    # Analytical gradient
    # Compute Euclidean gradient and overlap
    grad_euclidean, overlap = compute_full_gradient(
        U_mpo, gates_per_layer, layer_is_odd, max_bondim, gate_sites_per_layer, compute_overlap=True
    )

    projected = []
    for Umat, Zmat in zip(Vlist_start.reshape(-1, 4, 4), grad_euclidean.reshape(-1, 4, 4)):
        Z_proj = Zmat - Umat @ ((Umat.conj().T @ Zmat + Zmat.conj().T @ Umat) / 2)
        projected.append(Z_proj)
    grad_analytic = -jnp.stack(projected).reshape(Vlist_start.shape)


    epsilon = 1e-6
    grad_numeric = jnp.zeros_like(grad_analytic)
    if reference_is_normalized: const_F=2**int(n_sites/2)
    else: const_F=2**n_sites

    for i in range(Vlist_start.shape[0]):  # gate index
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        perturb = jnp.zeros_like(Vlist_start)
                        perturb = perturb.at[i, a, b, c, d].set(epsilon)

                        cost_plus, _ = get_riemannian_gradient_and_cost_function(
                            U_mpo, Vlist_start + perturb, n_sites, degree, n_repetitions, n_id_layers,
                            max_bondim, reference_is_normalized=True, hamiltonian=hamiltonian
                        )
                        cost_minus, _ = get_riemannian_gradient_and_cost_function(
                            U_mpo, Vlist_start - perturb, n_sites, degree, n_repetitions, n_id_layers,
                            max_bondim, reference_is_normalized=True, hamiltonian=hamiltonian
                        )

                        grad_numeric = grad_numeric.at[i, a, b, c, d].set((cost_plus - cost_minus) / (2 * epsilon))                        


    # Relative error
    err = jnp.linalg.norm(grad_analytic - grad_numeric) / jnp.linalg.norm(grad_numeric)
    print(f"[Gradient Check] Relative error: {err:.2e}")

    print("gate[0] (reshaped):", Vlist_start[0].reshape(2,2,2,2))
    print("gate_per_layer[0][0]:", gates_per_layer[0][0])
    print(f"gate_sites_per_layer[0][0]: {gate_sites_per_layer[0][0]}")
    print("Analytic grad[0]:", grad_analytic[0], grad_analytic[0].conj(), grad_numeric[0])
    print("Numeric grad[0]:", grad_numeric[0])
    print("Difference:", grad_analytic[0] - grad_numeric[0])

    print("Cost from get_riemannian_gradient_and_cost_function:", cost_plus)
    print("Overlap from compute_full_gradient:", overlap)
    print("Expected cost: ", 2 - 2 * overlap.real / const_F)

    print("‖grad_analytic‖ =", jnp.linalg.norm(grad_analytic))
    print("‖grad_numeric‖ =", jnp.linalg.norm(grad_numeric))


    return grad_numeric, grad_analytic


def compute_partial_derivatives_in_layer(
    gates_in_layer,
    layer_odd: bool,
    gate_sites: Sequence[Tuple[int,int]],
    upper_env_mpo,
    lower_env_mpo
):
    n = len(upper_env_mpo)
    # 1) Initialize left/right environments as before
    if layer_odd:
        i_mpo = n
        R, L = jnp.eye(1), jnp.eye(1)
    else:
        i_mpo = n-1
        R = jnp.einsum('abcd,ecbd->ae', upper_env_mpo[-1], lower_env_mpo[-1])
        L = jnp.einsum('abcd,acbe->de', upper_env_mpo[0],    lower_env_mpo[0])
    R_envs = [R]

    # 2) Build the list of *all* possible two-site windows in this layer
    if layer_odd:
        start = 0
    else:
        start = 1
    site_pairs = [(i, i+1) for i in range(start, n-1, 2)]

    # 3) Compute right environments for those gates *in reverse order*
    for gate, (i,j) in zip(reversed(gates_in_layer), reversed(site_pairs)):
        # identical to your old code, but you know (i,j) is one of the true gate sites
        A1, A2 = upper_env_mpo[i], upper_env_mpo[j]
        B1, B2 = lower_env_mpo[i], lower_env_mpo[j]
        if gate is not None:
            R = jnp.einsum('abcd,defg,cfhk,ihbj,jkel,gl->ai',
                        A1, A2, gate, B1, B2, R)
        else:
            R = jnp.einsum('abcd,cfhk,ihbj,gl->ai',
                        A1, A2, B1, B2, R)
        R_envs.append(R)
    R_envs = R_envs[::-1]

    # 4) Now do the left-to-right pass, computing a gradient *only* when (i,j)∈gate_sites
    grad = []
    L_current = L
    for idx, (gate, (i,j)) in enumerate(zip(gates_in_layer, site_pairs)):
        if (i,j) not in gate_sites:
            grad.append(jnp.zeros((2,2,2,2)))
            # no gate here in PXP → skip
            continue

        # build A1,A2,B1,B2 exactly as before:
        A1, A2 = upper_env_mpo[i], upper_env_mpo[i+1]
        B1, B2 = lower_env_mpo[i], lower_env_mpo[i+1]
        R_current = R_envs[idx+1]   # because R_envs[0] is the rightmost edge
        # update L_current if idx>0 …
        if idx>0:
            prev_gate, (pi,pj) = gates_in_layer[idx-1], site_pairs[idx-1]
            L_current = jnp.einsum('ai,abcd,defg,cfhk,ihbj,jkel->gl',
                                   L_current,
                                   upper_env_mpo[pi], upper_env_mpo[pj],
                                   prev_gate,
                                   lower_env_mpo[pi], lower_env_mpo[pj])
        # now full contraction
        res = jnp.einsum('ab,acde,efgh,bick,kjfl,hl->dgij',
                        L_current, A1, A2, B1, B2, R_current)
        grad.append(res)

    return jnp.stack(grad).conj()


def compute_full_gradient(U_mpo, gates_per_layer, layer_is_odd, max_bondim,
                          gate_sites_per_layer=None, compute_overlap=False):
    bottom_env, top_env = get_id_mpo(len(U_mpo)), U_mpo.copy()
    upper_envs, grad = [top_env.copy()], []
    merge_right_to_left = True  # Assume a left-canonicalized reference

    # Build upper environments
    for layer_idx in reversed(range(1, len(gates_per_layer))):
        gates = gates_per_layer[layer_idx]
        odd = layer_is_odd[layer_idx]
        gate_sites = gate_sites_per_layer[layer_idx] if gate_sites_per_layer is not None else None

        if merge_right_to_left:
            top_env = merge_and_truncate_mpo_and_layer_right_to_left(
                top_env, gates, odd_layer=odd, max_bondim=max_bondim, 
                **({"gate_sites": gate_sites} if gate_sites is not None else {}),
                layer_is_below=True
            )
        else:
            top_env = merge_and_truncate_mpo_and_layer_left_to_right(
                top_env, gates, odd_layer=odd, max_bondim=max_bondim, 
                **({"gate_sites": gate_sites} if gate_sites is not None else {}),
                layer_is_below=True
            )
        upper_envs.append(top_env.copy())
        merge_right_to_left = not merge_right_to_left
    upper_envs = upper_envs[::-1]

    # Build bottom environments and compute gradients
    for layer_idx in range(len(gates_per_layer)):
        if layer_idx > 0:
            prev_idx = layer_idx - 1
            gates = gates_per_layer[prev_idx]
            odd = layer_is_odd[prev_idx]
            gate_sites = gate_sites_per_layer[prev_idx] if gate_sites_per_layer is not None else None

            if merge_right_to_left:
                bottom_env = merge_and_truncate_mpo_and_layer_right_to_left(
                    bottom_env, gates, odd_layer=odd, max_bondim=max_bondim, 
                    **({"gate_sites": gate_sites} if gate_sites is not None else {}),
                    layer_is_below=False
                )
            else:
                bottom_env = merge_and_truncate_mpo_and_layer_left_to_right(
                    bottom_env, gates, odd_layer=odd, max_bondim=max_bondim, 
                    **({"gate_sites": gate_sites} if gate_sites is not None else {}),
                    layer_is_below=False
                )
            merge_right_to_left = not merge_right_to_left

        grad += list(compute_partial_derivatives_in_layer(
            gates_per_layer[layer_idx], layer_is_odd[layer_idx], gate_sites_per_layer[layer_idx], upper_envs[layer_idx], bottom_env
        ))
    

    grad = jnp.asarray(grad)

    # Apply normalization consistent with cost
    if compute_overlap:
        overlap = jnp.einsum('abcd,abcd->', grad[0].conj(), gates_per_layer[0][0])
        d = 2 ** len(U_mpo)
        grad = grad / d
        return grad, overlap
    else:
        d = 2 ** len(U_mpo)
        grad = grad / d
        return grad


def get_riemannian_gradient_and_cost_function(U_mpo, Vlist_TN, n_sites, degree, n_repetitions, n_id_layers,
                                              max_bondim, reference_is_normalized, hamiltonian, gates_per_layer, gate_sites_per_layer, layer_is_odd):
    #if hamiltonian=='fermi-hubbard-1d': n_sites = 2*n_orbitals
    #elif hamiltonian=='molecular': n_sites = n_orbitals

    grad, overlap = compute_full_gradient(U_mpo, gates_per_layer, layer_is_odd, max_bondim, gate_sites_per_layer, compute_overlap=True)
    # Get Riemannian gradient from Euclidean gradient
    vlist_reshaped = Vlist_TN.reshape((Vlist_TN.shape[0],4,4))

    if grad.shape != Vlist_TN.shape:
        raise ValueError(f"Mismatch in shape: grad={grad.shape}, Vlist_TN={Vlist_TN.shape}")
    grad_reshaped = grad.reshape(vlist_reshaped.shape)
    projected_grad = - project_unitary_tangent_vectorized(vlist_reshaped, grad_reshaped, True)

    # Get Frobenius norm and Hilbert-Schmidt test from overlap -> normalization holds for normalized reference!
    if reference_is_normalized: const_F=2**int(n_sites/2)
    else: const_F=2**n_sites
    cost_F = 2 - 2*overlap.real/const_F  # Frobenius norm

    return cost_F, projected_grad


def get_riemannian_gradient_and_cost_function_general(U_ref, Vlist_TN, max_bondim, reference_is_normalized):
    """
    Compute cost and gradient for general gate sequences (not assuming brickwall layout).
    
    Parameters
    ----------
    U_ref : MPO
        Reference unitary in MPO form.
    Vlist_TN : np.ndarray
        Gate list of shape (N_gates, 2, 2, 2, 2).
    max_bondim : int
        Max bond dimension allowed during contraction.
    reference_is_normalized : bool
        If True, adjusts normalization factor for overlap.
    
    Returns
    -------
    cost_F : float
        Frobenius norm cost.
    projected_grad : np.ndarray
        Riemannian gradient in the tangent space.
    """

    # === 1. Forward pass: build U_ansatz ===
    n_sites = U_ref.nsites
    U0 = get_id_mpo(n_sites)
    U_ansatz = contract_gate_list_with_mpo(U0, Vlist_TN, max_bond=max_bondim)

    # === 2. Compute overlap (Hilbert-Schmidt inner product) ===
    overlap = compute_overlap(U_ref, U_ansatz)

    # === 3. Cost: Frobenius norm distance ===
    norm_factor = 2**(n_sites if not reference_is_normalized else n_sites // 2)
    cost_F = 2 - 2 * overlap.real / norm_factor

    # === 4. Compute Euclidean gradient (finite-diff placeholder or autodiff later) ===
    # For now, just use a dummy placeholder — you'll plug in real gradient logic next
    grad = jnp.zeros_like(Vlist_TN)  # TODO: replace with autodiff or analytical gradient

    # === 5. Project gradient to tangent space (Riemannian) ===
    vlist_flat = Vlist_TN.reshape((-1, 4, 4))
    grad_flat = grad.reshape((-1, 4, 4))
    projected_grad = -project_unitary_tangent_vectorized(vlist_flat, grad_flat, use_TN=True)

    return cost_F, projected_grad


def get_cosine_fidelity_cost_function(U_mpo, Vlist_TN, n_sites, degree, n_repetitions, n_id_layers,
                                        max_bondim, reference_is_normalized, hamiltonian, n_layers=None):
    total_gates = Vlist_TN.shape[0]
    N_odd_gates = n_sites // 2
    N_even_gates = (n_sites - 1) // 2
    gates_per_rep = N_odd_gates + N_even_gates

    if n_layers is not None:
        gates_per_layer, gate_sites_per_layer, layer_is_odd = get_gates_per_layer(
            Vlist_TN, n_sites=n_sites, n_layers=n_layers,
            degree=degree, hamiltonian=hamiltonian, n_id_layers=n_id_layers
        )
    elif total_gates < gates_per_rep * n_repetitions:
        gates_per_layer_pattern = [N_odd_gates, N_even_gates]
        layer_sizes = []
        i = 0
        while sum(layer_sizes) + gates_per_layer_pattern[i % 2] <= total_gates:
            layer_sizes.append(gates_per_layer_pattern[i % 2])
            i += 1
        max_possible_layers = len(layer_sizes)
        gates_per_layer, gate_sites_per_layer, layer_is_odd = get_gates_per_layer(
            Vlist_TN, n_sites=n_sites, n_layers=max_possible_layers,
            degree=degree, hamiltonian=hamiltonian, n_id_layers=n_id_layers
        )
    else:
        gates_per_layer, gate_sites_per_layer, layer_is_odd = get_gates_per_layer(
            Vlist_TN, n_sites=n_sites, degree=degree, n_repetitions=n_repetitions,
            hamiltonian=hamiltonian, n_id_layers=n_id_layers
        )

    # Build the MPO from the circuit
    mpo_out = contract_layers_of_swap_network_with_mpo(
        get_id_mpo(n_sites), gates_per_layer, layer_is_odd,
        layer_is_left=True, max_bondim=max_bondim, get_norm=False
    )

    # Convert both MPOs to MPS
    mps_U = convert_mpo_to_mps(U_mpo)
    mps_W = convert_mpo_to_mps(mpo_out)

    # Normalize both MPS
    mps_U_nrmd, norm_U = get_left_canonical_mps(mps_U, normalize=False, get_norm=True)
    mps_U_nrmd = [t / norm_U for t in mps_U_nrmd]

    mps_W_nrmd, norm_W = get_left_canonical_mps(mps_W, normalize=False, get_norm=True)
    mps_W_nrmd = [t / norm_W for t in mps_W_nrmd]

    # Compute cosine cost = 1 - fidelity
    overlap = inner_product_mps(mps_U_nrmd, mps_W_nrmd).real
    cost_cosine = 1 - overlap ** 2

    # Compute gradient (still Frobenius-based)
    grad = compute_full_gradient(U_mpo, gates_per_layer, layer_is_odd,
                                    max_bondim=max_bondim, compute_overlap=False)

    vlist_reshaped = Vlist_TN.reshape((Vlist_TN.shape[0], 4, 4))
    grad_reshaped = grad.reshape(vlist_reshaped.shape)
    projected_grad = -project_unitary_tangent_vectorized(vlist_reshaped, grad_reshaped, True)

    return cost_cosine, projected_grad


def fully_contract_swap_network_mpo(
        Vlist_TN, U_mpo, degree, n_repetitions, n_id_layers, n_layers,
        max_bondim, hamiltonian):
    
    n_sites = len(U_mpo)
    
    gates_per_layer, gate_sites_per_layer, layer_is_odd = get_gates_per_layer(
        Vlist_TN, n_sites, degree, n_repetitions, n_layers, n_id_layers, hamiltonian)
    
    layer_is_left = True
    mpo_res = contract_layers_of_swap_network_with_mpo(U_mpo, gates_per_layer, layer_is_odd,
                                                       layer_is_left, max_bondim)
    trace = fully_contract_mpo(mpo_res)
    return trace


# ################################################################################
    
# def merge_and_truncate_mpo_and_layer_right_to_left(mpo_init, gates_in_layer, odd_layer=True, 
#                                                    max_bondim=128, layer_is_below=True):
#     '''
#     layer_is_below corresponds to layer_is_left
#     '''
#     n_spin_orbitals=len(mpo_init)
#     if not odd_layer:  # For even layers
#         Q, R = canonicalize_local_tensor(mpo_init[-1], left=False)
#         mpo_res = [Q]; i_mpo_init=n_spin_orbitals-2
#     else:
#         mpo_res=[]; i_mpo_init=n_spin_orbitals-1

#     while i_mpo_init-1>=0:
#         mpo1, mpo2 = mpo_init[i_mpo_init-1], mpo_init[i_mpo_init]
#         if i_mpo_init==n_spin_orbitals-1: mpo2_R = mpo2
#         else: mpo2_R = jnp.einsum('iabj,jk->iabk', mpo2, R)  # Merge the R tensor in last local mpo
#         gate = gates_in_layer[int((i_mpo_init-1)/2)]
#         merged_T = merge_two_mpos_and_gate(gate, mpo1, mpo2_R, gate_is_left=layer_is_below)  # Merge gate and local tensor pair
#         T1, T2 = split_tensor_into_half_canonical_mpo_pair(merged_T, canonical_mode='right', max_bondim=max_bondim)
#         if i_mpo_init-1==0:
#             mpo_res += [T2, T1]  # T2, T1
#         else: 
#             Q, R = canonicalize_local_tensor(T1, left=False)
#             mpo_res += [T2, Q]
#         i_mpo_init -= 2
    
#     if i_mpo_init==0: 
#             mpo_R = jnp.einsum('iabj,jk->iabk', mpo_init[0], R)
#             mpo_res += [mpo_R] 

#     mpo_res = mpo_res[::-1]  # Reverse the order of MPO
#     return mpo_res

# def merge_and_truncate_mpo_and_layer_left_to_right(mpo_init, gates_in_layer, odd_layer=False, 
#                                                    max_bondim=128, layer_is_below=True):
#     """
#     Function to merge and compress a layer of two-qubit gates with an MPO.
#     The sweep is done from left to right using RQ decomposition.
#     """
#     n_spin_orbitals=len(mpo_init)
#     if not odd_layer:  # For even layers
#         Q, R = canonicalize_local_tensor(mpo_init[0], left=True)
#         mpo_res = [Q]; i_mpo_init=1
#     else:
#         mpo_res=[]; i_mpo_init=0

#     while i_mpo_init+1<n_spin_orbitals:
#         mpo1, mpo2 = mpo_init[i_mpo_init:i_mpo_init+2]
#         if i_mpo_init==0: mpo1_R = mpo1
#         else: mpo1_R = jnp.einsum('ij,jabk->iabk', R, mpo1)  # Merge the R tensor in first local mpo
#         gate = gates_in_layer[int(i_mpo_init/2)]
#         merged_T = merge_two_mpos_and_gate(gate, mpo1_R, mpo2, gate_is_left=layer_is_below)  # Merge gate and local tensor pair
#         T1, T2 = split_tensor_into_half_canonical_mpo_pair(merged_T, canonical_mode='left', max_bondim=max_bondim)
#         if i_mpo_init+1==n_spin_orbitals-1:
#             mpo_res += [T1, T2]
#         else: 
#             Q, R = canonicalize_local_tensor(T2, left=True)
#             mpo_res += [T1, Q]
#         i_mpo_init += 2

#     if i_mpo_init==n_spin_orbitals-1: 
#         mpo_R = jnp.einsum('ij,jabk->iabk', R, mpo_init[-1])
#         mpo_res += [mpo_R]        

#     return mpo_res



# def compute_partial_derivatives_in_layer(gates_in_layer, layer_odd, upper_env_mpo, lower_env_mpo):
#     if layer_odd:  # If the layer is odd, the edge qubits have a gate acting on it
#         i_mpo = len(upper_env_mpo)
#         R, L = jnp.eye(1), jnp.eye(1)
#     else:  # Otherwise, the edge environments are given by the local MPOs
#         i_mpo = len(upper_env_mpo)-1
#         R = jnp.einsum('abcd,ecbd->ae', upper_env_mpo[-1], lower_env_mpo[-1])
#         L = jnp.einsum('abcd,acbe->de', upper_env_mpo[0], lower_env_mpo[0])
#     R_envs = [R.copy()]

#     # Compute all right environments (go from right to left)
#     for gate in reversed(gates_in_layer[1:]):
#         A1, A2, B1, B2 = *upper_env_mpo[i_mpo-2:i_mpo], *lower_env_mpo[i_mpo-2:i_mpo]
#         R = jnp.einsum('abcd,defg,cfhk,ihbj,jkel,gl->ai', A1, A2, gate, B1, B2, R)
#         R_envs.append(R.copy())
#         i_mpo -= 2
#     R_envs = R_envs[::-1]

#     # Compute now all partial derivatives for an odd layer (go from left to right)
#     grad = jnp.empty_like(gates_in_layer)
#     if layer_odd: i_mpo, i_env_mpo = 0, 0
#     else: i_mpo, i_env_mpo = 1, 1
#     for cut_out_gate in range(len(gates_in_layer)):
#         # Get local MPO tensors of lower and upper environments
#         A1, A2, B1, B2 = *upper_env_mpo[i_mpo:i_mpo+2], *lower_env_mpo[i_mpo:i_mpo+2]
#         i_mpo += 2

#         # Compute R, L
#         if cut_out_gate>0:  # R,L for the very left gate are already computed
#             # Current R
#             R = R_envs[cut_out_gate]  
#             # Current L
#             L = jnp.einsum('ai,abcd,defg,cfhk,ihbj,jkel->gl', L, *upper_env_mpo[i_env_mpo:i_env_mpo+2], 
#                            gates_in_layer[cut_out_gate-1], *lower_env_mpo[i_env_mpo:i_env_mpo+2])
#             i_env_mpo += 2

#         # Contract everything
#         res = jnp.einsum('ab,acde,efgh,bick,kjfl,hl->dgij', L, A1, A2, B1, B2, R)
#         grad = grad.at[cut_out_gate].set(res)
#     return grad.conj()


# def compute_full_gradient(U_mpo, gates_per_layer, layer_is_odd, max_bondim, compute_overlap=False):
#     # Initialize edge environments
#     bottom_env, top_env = get_id_mpo(len(U_mpo)), U_mpo.copy()
#     upper_envs, grad = [top_env.copy()], []
#     merge_right_to_left = True  # Assume a left canonicalized reference

#     # Compute all upper environments (go from top to bottom)
#     for gates, odd in zip(reversed(gates_per_layer[1:]), reversed(layer_is_odd[1:])):
#         # Merge all layers up to the last one (from left to right) of the swap network into the initial id MPO
#         if merge_right_to_left: top_env = merge_and_truncate_mpo_and_layer_right_to_left(
#             top_env, gates, odd, max_bondim, layer_is_below=True)
#         else: top_env = merge_and_truncate_mpo_and_layer_left_to_right(
#             top_env, gates, odd, max_bondim, layer_is_below=True)
#         upper_envs.append(top_env.copy())
#         merge_right_to_left = not merge_right_to_left
#     upper_envs = upper_envs[::-1]

#     # Now compute the gradient
#     for layer in range(len(gates_per_layer)):
#         if layer>0: 
#             #odd = layer_is_odd[layer-1]
#             if merge_right_to_left: bottom_env = merge_and_truncate_mpo_and_layer_right_to_left(
#                 bottom_env, gates_per_layer[layer-1], layer_is_odd[layer-1], max_bondim, layer_is_below=False)
#             else: bottom_env = merge_and_truncate_mpo_and_layer_left_to_right(
#                 bottom_env, gates_per_layer[layer-1], layer_is_odd[layer-1], max_bondim, layer_is_below=False)
#             merge_right_to_left = not merge_right_to_left

#         # Compute the partial derivatives in this layer
#         grad += list(compute_partial_derivatives_in_layer(gates_per_layer[layer], layer_is_odd[layer], upper_envs[layer], bottom_env))

#     grad = jnp.asarray(grad)
#     if compute_overlap: 
#         overlap = jnp.einsum('abcd,abcd->', grad[0].conj(), gates_per_layer[0][0])  # Compute tr(U^\dagger W)
#         return grad, overlap 
#     else: return grad


# def get_riemannian_gradient_and_cost_function(U_mpo, Vlist_TN, n_sites, degree, n_repetitions, n_id_layers,
#                                               max_bondim, reference_is_normalized, hamiltonian):
#     #if hamiltonian=='fermi-hubbard-1d': n_sites = 2*n_orbitals
#     #elif hamiltonian=='molecular': n_sites = n_orbitals
#     gates_per_layer, _, layer_is_odd = get_gates_per_layer(Vlist_TN, n_sites, degree, n_repetitions, 
#                                                         n_id_layers=n_id_layers, hamiltonian=hamiltonian)
#     grad, overlap = compute_full_gradient(U_mpo, gates_per_layer, layer_is_odd, max_bondim, compute_overlap=True)

#     # Get Riemannian gradient from Euclidean gradient
#     vlist_reshaped = Vlist_TN.reshape((Vlist_TN.shape[0],4,4))
#     grad_reshaped = grad.reshape(vlist_reshaped.shape)
#     projected_grad = - project_unitary_tangent_vectorized(vlist_reshaped, grad_reshaped, True)

#     # Get Frobenius norm and Hilbert-Schmidt test from overlap -> normalization holds for normalized reference!
#     if reference_is_normalized: const_F=2**int(n_sites/2)
#     else: const_F=2**n_sites
#     cost_F = 2 - 2*overlap.real/const_F  # Frobenius norm

#     return cost_F, projected_grad


# def get_riemannian_gradient_and_cost_function(U_mpo, Vlist_TN, n_sites, degree, n_repetitions, n_id_layers,
#                                               max_bondim, reference_is_normalized, hamiltonian, n_layers=None):

#     total_gates = Vlist_TN.shape[0]
#     N_odd_gates = n_sites // 2
#     N_even_gates = (n_sites - 1) // 2
#     gates_per_rep = N_odd_gates + N_even_gates

#     if n_layers is not None:
#         gates_per_layer, layer_is_odd = get_gates_per_layer(
#             Vlist_TN, n_sites=n_sites, n_layers=n_layers,
#             degree=degree, hamiltonian=hamiltonian, n_id_layers=n_id_layers
#         )

#     elif total_gates < gates_per_rep * n_repetitions:
#         gates_per_layer_pattern = [N_odd_gates, N_even_gates]
#         layer_sizes = []
#         i = 0
#         while sum(layer_sizes) + gates_per_layer_pattern[i % 2] <= total_gates:
#             layer_sizes.append(gates_per_layer_pattern[i % 2])
#             i += 1
#         max_possible_layers = len(layer_sizes)
#         gates_per_layer, layer_is_odd = get_gates_per_layer(
#             Vlist_TN, n_sites=n_sites, n_layers=max_possible_layers,
#             degree=degree, hamiltonian=hamiltonian, n_id_layers=n_id_layers
#         )

#     else:
#         gates_per_layer, layer_is_odd = get_gates_per_layer(
#             Vlist_TN, n_sites=n_sites, degree=degree, n_repetitions=n_repetitions,
#             hamiltonian=hamiltonian, n_id_layers=n_id_layers
#         )

#     grad, overlap = compute_full_gradient(U_mpo, gates_per_layer, layer_is_odd, max_bondim, compute_overlap=True)

#     vlist_reshaped = Vlist_TN.reshape((Vlist_TN.shape[0], 4, 4))
#     grad_reshaped = grad.reshape(vlist_reshaped.shape)
#     projected_grad = - project_unitary_tangent_vectorized(vlist_reshaped, grad_reshaped, True)

#     const_F = 2**(n_sites // 2) if reference_is_normalized else 2**n_sites
#     cost_F = 2 - 2 * overlap.real / const_F

#     return cost_F, projected_grad

  
# def compute_partial_derivatives_in_layer(gates_in_layer, layer_odd, upper_env_mpo, lower_env_mpo):
#     if layer_odd:
#         # --- use the actual right-bond dimension of the current MPO ---
#         bond_dim_R = upper_env_mpo[-1].shape[3]        # A2.h of the last tensor
#         R = jnp.eye(bond_dim_R)                        # shape (h, h)
#         L = jnp.eye(1)                                 # left edge is still rank-1
#         i_mpo = len(upper_env_mpo)
#     else:
#         i_mpo = len(upper_env_mpo) - 1
#         R = jnp.einsum('abcd,ecbd->ae', upper_env_mpo[-1], lower_env_mpo[-1])
#         L = jnp.einsum('abcd,acbe->de', upper_env_mpo[0],  lower_env_mpo[0])
#     R_envs = [R.copy()]

#     gates_in_layer = jnp.asarray(gates_in_layer)

#     # Compute all right environments (go from right to left)
#     for gate in reversed(gates_in_layer[1:]):
#         G = gate.copy()

#         A1, A2, B1, B2 = *upper_env_mpo[i_mpo-2:i_mpo], *lower_env_mpo[i_mpo-2:i_mpo]
#         R = jnp.einsum('abcd,defg,cfhk,ihbj,jkel,gl->ai', A1, A2, gate, B1, B2, R)
#         R_envs.append(R.copy())
#         i_mpo -= 2
#     R_envs = [None] + R_envs[::-1]

#     # Compute now all partial derivatives for an odd layer (go from left to right)
#     # grad = jnp.empty_like(gates_in_layer)
#     grad = [jnp.empty_like(g) for g in gates_in_layer]

#     if layer_odd: i_mpo, i_env_mpo = 0, 0
#     else: i_mpo, i_env_mpo = 1, 1
#     for cut_out_gate in range(len(gates_in_layer)):
#         # Get local MPO tensors of lower and upper environments
#         A1, A2, B1, B2 = *upper_env_mpo[i_mpo:i_mpo+2], *lower_env_mpo[i_mpo:i_mpo+2]
#         i_mpo += 2

#         # Compute R, L
#         if cut_out_gate>0:  # R,L for the very left gate are already computed
#             # Current R
#             R = R_envs[cut_out_gate+2]  
#             # Current L
#             L = jnp.einsum('ai,abcd,defg,cfhk,ihbj,jkel->gl', L, *upper_env_mpo[i_env_mpo:i_env_mpo+2], 
#                            gates_in_layer[cut_out_gate-1], *lower_env_mpo[i_env_mpo:i_env_mpo+2])
#             i_env_mpo += 2
#         else:
#         # --- first gate: match A2’s right bond, not the global one ---
#             R = jnp.eye(A2.shape[3])  

#         # Contract everything
#         print(f"[Gate {cut_out_gate}]")
#         print("A1", A1.shape)
#         print("A2", A2.shape)
#         print("B1", B1.shape)
#         print("B2", B2.shape)
#         print("R", R.shape)
#         current_gate = gates_in_layer[cut_out_gate]
#         print("gate", current_gate.shape)
        
#         assert A2.shape[3] == R.shape[0], (
#             f"Mismatch in final contraction:\n"
#             f"A2.shape={A2.shape}, R.shape={R.shape}, cut_out_gate={cut_out_gate}, "
#             f"i_mpo={i_mpo}, i_env_mpo={i_env_mpo}"
#         )

#         res = jnp.einsum('ab,acde,efgh,bick,kjfl,hl->dgij', L, A1, A2, B1, B2, R)
#         # grad = grad.at[cut_out_gate].set(res)
#         grad[cut_out_gate] = res

#     # return grad.conj()
#     return jnp.stack([g.conj() for g in grad])

  
# def layer_acts_on_right_edge(site_list, n_sites):
#     """True iff the layer contains a gate whose right qubit is n_sites-1."""
#     return bool(site_list and site_list[-1][1] == n_sites - 1)
