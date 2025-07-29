import numpy as np
import jax.numpy as jnp
from jax import config as c
c.update("jax_enable_x64", True)
from scipy.optimize import curve_fit

def get_trotter_scaling(ts, cost):
    def _func(x, a, b):
        return a + b*x
    x = np.log(np.asarray(ts))
    y = np.log(np.asarray(cost))
    popt, _ = curve_fit(_func, x, y)
    func = lambda cost: jnp.asarray([jnp.exp(popt[0] + popt[1]*x_) for x_ in x])
    return jnp.exp(x), func, popt


def get_identity_two_qubit_tensors(n, use_TN=True):
    """
    Generate n pairs of random tensors that correspond to 
    a two-qubit gates.
    ---|‾|---
       | |
    ---|_|---
    """
    matrices = jnp.asarray([jnp.eye(4) for _ in range(n)])
    if use_TN: tn = matrices.reshape((n,2,2,2,2))
    else: tn = matrices
    return jnp.asarray(tn)


def get_identity_layers(n_sites, n_layers, first_layer_odd, use_TN=True):
    """
    Generate n_layers layers with two-qubit gates expressed
    as tensors --> alternate between odd and even layers.
    """
    layers = []
    odd = first_layer_odd
    N_odd_gates, N_even_gates = int(n_sites/2), int(n_sites/2)  # Number of gates per layer
    if n_sites%2==0: N_even_gates -= 1
    for _ in range(n_layers):
        n_gates=N_odd_gates if odd else N_even_gates
        layers += list(get_identity_two_qubit_tensors(n_gates, use_TN))
        odd = not odd
    return jnp.asarray(layers)     


def swap(chain, ip,iq):
    # Swap positions ip and iq in a linear chain
    chain = jnp.asarray(chain)
    dummy = chain[ip]
    chain = chain.at[ip].set(chain[iq])
    chain = chain.at[iq].set(dummy)
    return chain

def retract_unitary_(V_TN, eta):
    """
    Retraction for unitary matrices, with tangent direction represented as anti-symmetric matrices.
    """
    V = V_TN.reshape((4,4))
    eta = jnp.reshape(eta, jnp.shape(V))
    dvlist = V @ real_to_antisymm(eta)
    return jnp.asarray(polar_decomp(V + dvlist)[0].reshape((2,2,2,2)))


def polar_decomp(a):
    """
    Perform a polar decomposition of a matrix: ``a = u p``,
    with `u` unitary and `p` positive semidefinite.
    """
    u, s, vh = jnp.linalg.svd(a)
    return u @ vh, (vh.conj().T * s) @ vh


def svd_retraction(X, W):
    u, s, vh = jnp.linalg.svd(X + W)
    return u @ vh

def qr_retraction(X, W):
    q, _ = jnp.linalg.qr(X + W)
    return q


def retract_unitary(V_TN, eta, use_TN=False):
    """
    Retraction for unitary matrices, with tangent direction represented as anti-symmetric matrices.
    -> because we mapped the gradient to a real vector, we need to map it back!
    """
    V = V_TN.reshape((4,4))
    eta = jnp.reshape(eta, (4, 4))
    res = svd_retraction(V, eta)
    if use_TN: res = res.reshape((2,2,2,2))
    return res


def real_to_antisymm(r):
    """
    Map a real-valued square matrix to an anti-symmetric matrix of the same dimension.
    """
    return 0.5*(r - r.T) + 0.5j*(r + r.T)


def project_unitary_tangent(u, z, use_TN=False):
    """
    Project `z` onto the tangent plane at the unitary matrix `u`.
    """
    # formula remains valid for `u` an isometry (element of the Stiefel manifold)
    if use_TN:
        shape0 = u.shape; shape = (4,4)
        u_ = u.reshape(shape); z_ = z.reshape(shape)
        res = 0.5*z_ - 0.5 * u_@z_.conj().T@u_
        #res = res.reshape(shape0)
    else:
        res = 0.5*z - 0.5 * u@z.conj().T@u
    return res

def project_unitary_tangent_vectorized(u, z, use_TN=False):
    """
    Project `z` onto the tangent plane at the unitary matrix `u`.
    Eq. (28)
    Vectorized to handle a list of matrices `u` and `z`.
    """
    def _proj_matrices(U, Z):
        return Z - jnp.einsum(
            'ijk,ikl->ijl', U, symm_vectorized(
                jnp.einsum('ijk,ijm->ikm', U.conj(), Z)))
    if use_TN:
        shape0 = u.shape; shape = (shape0[0],4,4)
        u_ = u.reshape(shape); z_ = z.reshape(shape)
        res = _proj_matrices(u_,z_).reshape(shape0)
    else:
        res = _proj_matrices(u,z)
    return res

def symm_vectorized(w):
    return 0.5 * (w + jnp.einsum('ijk->ikj', w.conj()))

def inner_product(V, X, Y, use_TN=False):
    '''
    Take the inner product of vector $X$ and vector $Y$ in point $V$.
    $\langle X, Y \rangle_V = Tr(X^\dagger Y)$.
    '''
    if use_TN: V_reshaped = V.reshape((4, 4))
    else: V_reshaped = V.copy()
    X_reshaped, Y_reshaped = X.reshape(V_reshaped.shape), Y.reshape(V_reshaped.shape)
    res = jnp.trace(X_reshaped.conj().T @Y_reshaped)
    return res

def apply_gate_to_mpo(mpo, gate_tensor, site=None, contract=True, max_bond=None):
    """
    Apply a 2-qubit gate tensor to a pair of adjacent sites in an MPO.

    Parameters
    ----------
    mpo : quimb.tensor.MatrixProductOperator
        The MPO to apply the gate to.
    gate_tensor : np.ndarray
        A (2,2,2,2) unitary gate tensor.
    site : int or None
        If None, applies sequentially (assumes gate #k acts on sites k, k+1).
        If int, applies to (site, site+1).
    contract : bool
        Whether to contract the result back into MPO form.
    max_bond : int or None
        Maximum bond dimension to keep after applying.

    Returns
    -------
    mpo : quimb.tensor.MatrixProductOperator
        The updated MPO.
    """
    import quimb.tensor as qtn

    if site is None:
        # infer default site based on current gate index
        site = getattr(mpo, '_last_gate_site', 0)
        mpo._last_gate_site = site + 1  # auto-increment for next gate

    gate = qtn.Tensor(
        data=gate_tensor,
        inds=(f'k{site}', f'k{site+1}', f'b{site}', f'b{site+1}'),
        tags={'GATE'}
    )

    tn = mpo.gate_inds_with_tn(
        gate,
        gate_inds_inner=(f'b{site}', f'b{site+1}'),
        gate_inds_outer=(f'k{site}', f'k{site+1}'),
        inds=(f'b{site}', f'b{site+1}'),
    )

    return qtn.MatrixProductOperator.from_TN(tn, max_bond=max_bond)
