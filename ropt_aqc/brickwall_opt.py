import os

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax import config as c
c.update("jax_enable_x64", True)

from .adam import RieADAM
from .util import project_unitary_tangent, retract_unitary, inner_product
from .tn_brickwall_methods import get_riemannian_gradient_and_cost_function, get_cosine_fidelity_cost_function, get_riemannian_gradient_and_cost_function_general, check_full_gradient
from ropt_aqc.brickwall_circuit import get_gates_per_layer

fig_dir = "/home/b/aag/ropt-aqc/production/Figures"

def optimize_swap_network_circuit_RieADAM(config, U, Vlist_start):
    """
    Optimize the quantum gates in a swap network layout to approximate
    the unitary matrix `U` using a Riemannian ADAM optimizer.
    Vlist_start is given in the form of tensors or matrices.
    """
    
    assert(Vlist_start.shape[1:]==(2,2,2,2))
    gates_per_layer, gate_sites_per_layer, layer_is_odd = get_gates_per_layer(
    Vlist_start,
    n_sites=config['n_sites'],
    degree=config['degree'],
    n_repetitions=config['n_repetitions'],
    n_id_layers=config['n_id_layers'],
    hamiltonian=config['hamiltonian']
)

    f_df = lambda vlist: (
        lambda gpl, gspl, lio: get_riemannian_gradient_and_cost_function(
            U, vlist, config['n_sites'], config['degree'], config['n_repetitions'],
            config['n_id_layers'], config['max_bondim'], config['normalize_reference'],
            config['hamiltonian'],
            gates_per_layer=gpl, gate_sites_per_layer=gspl, layer_is_odd=lio
        )
    )(*get_gates_per_layer(
        vlist, n_sites=config['n_sites'], degree=config['degree'],
        n_repetitions=config['n_repetitions'], n_id_layers=config['n_id_layers'],
        hamiltonian=config['hamiltonian']
    ))
    # Define retraction, projection, and inner product
    _retract = lambda v, eta: retract_unitary(v, eta, use_TN=True)
    _project = lambda u,z: project_unitary_tangent(u,z, True)
    # _project = lambda u, z: z
    retract, project = vmap(_retract), vmap(_project)

    # Set up the optimizer
    _metric = lambda v,x,y: inner_product(v,x,y,True)
    metric = vmap(_metric)
    opt = RieADAM(maxiter=config['n_iter'], lr=float(config['lr']), beta_1=float(config['beta_1']), beta_2=float(config['beta_2']))

    Vlist, neval, err_iter = opt.minimize(function=f_df, initial_point=Vlist_start,
                                          retract=retract, projection=project, metric=metric)


    err_iter1 = jnp.asarray(err_iter)
    # Cost function: Frobenius norm
    err_init = err_iter[0]
    err_opt = jnp.min(jnp.asarray(err_iter))
    err_end = err_iter[-1]

    print(f"err_init: {err_init}")
    print(f"err_end after {neval} iterations: {err_end}")
    print(f"err_opt: {err_opt}")
    print(f"err_init/err_opt: {err_init/err_opt}")

    _ = plot_loss(config, err_iter, err_opt, save=True)
                
    return Vlist, err_iter


# def optimize_swap_network_circuit_RieADAM(config, U, Vlist_start):
#     """
#     Optimize the quantum gates in a swap network layout to approximate
#     the unitary matrix `U` using a Riemannian ADAM optimizer.
#     Vlist_start is given in the form of tensors or matrices.
#     """
    
#     assert(Vlist_start.shape[1:]==(2,2,2,2))
#     f_df = lambda vlist: get_riemannian_gradient_and_cost_function(
#         U, vlist, config['n_sites'], config['degree'], config['n_repetitions'], config['n_id_layers'], 
#         config['max_bondim'], config['normalize_reference'], config['hamiltonian'])

#     # Define retraction, projection, and inner product
#     _retract = lambda v, eta: retract_unitary(v, eta, use_TN=True)
#     _project = lambda u,z: project_unitary_tangent(u,z, True)
#     retract, project = vmap(_retract), vmap(_project)
    
#     # Set up the optimizer
#     _metric = lambda v,x,y: inner_product(v,x,y,True)
#     metric = vmap(_metric)
#     opt = RieADAM(maxiter=config['n_iter'], lr=float(config['lr']))
#     Vlist, neval, err_iter = opt.minimize(function=f_df, initial_point=Vlist_start,
#                                           retract=retract, projection=project, metric=metric)


#     err_iter1 = jnp.asarray(err_iter)
#     # Cost function: Frobenius norm
#     err_init = err_iter[0]
#     err_opt = jnp.min(jnp.asarray(err_iter))
#     err_end = err_iter[-1]

#     print(f"err_init: {err_init}")
#     print(f"err_end after {neval} iterations: {err_end}")
#     print(f"err_opt: {err_opt}")
#     print(f"err_init/err_opt: {err_init/err_opt}")

#     _ = plot_loss(config, err_iter, err_opt, save=True)
                
#     return Vlist, err_iter


def optimize_swap_fidelity_circuit_RieADAM(config, U, Vlist_start, n_layers=None):
    """
    Optimize the quantum gates in a swap network layout to approximate
    the unitary matrix `U` using a Riemannian ADAM optimizer.
    Vlist_start is given in the form of tensors or matrices.
    """
    
    assert(Vlist_start.shape[1:]==(2,2,2,2))
    f_df = lambda vlist: get_cosine_fidelity_cost_function(
        U, vlist, config['n_sites'], config['degree'], config['n_repetitions'], config['n_id_layers'], 
        config['max_bondim'], config['normalize_reference'], config['hamiltonian'], n_layers=n_layers)

    # Define retraction, projection, and inner product
    _retract = lambda v, eta: retract_unitary(v, eta, use_TN=True)
    _project = lambda u,z: project_unitary_tangent(u,z, True)

    retract, project = vmap(_retract), vmap(_project)
    
    # Set up the optimizer
    _metric = lambda v,x,y: inner_product(v,x,y,True)
    metric = vmap(_metric)
    opt = RieADAM(maxiter=config['n_iter'], lr=float(config['lr']))
    Vlist, neval, err_iter = opt.minimize(function=f_df, initial_point=Vlist_start,
                                          retract=retract, projection=project, metric=metric)

    cost_iter = jnp.asarray(err_iter)
    cost_init = cost_iter[0]
    cost_opt = jnp.min(cost_iter)
    cost_end = cost_iter[-1]

    print(f"cost_init (1 - fidelity²): {cost_init}")
    print(f"cost_end after {neval} iterations: {cost_end}")
    print(f"cost_opt (min value): {cost_opt}")
    print(f"Initial fidelity²: {1 - cost_init}")
    print(f"Final fidelity²:   {1 - cost_end}")

    _ = plot_loss(config, cost_iter, cost_opt, save=True)
                
    return Vlist, err_iter


def plot_loss(config, err_iter, err_opt, save=False):
    # Visualize optimization progress
    points = jnp.arange(len(err_iter))
    err_iter = jnp.asarray(err_iter)

    label = 'err_init={:.2e}\nerr_end={:.2e}\nerr_opt={:.2e}\nerr_init/err_opt={:.4f}'.format(
        err_iter[0], err_iter[-1], err_opt, err_iter[0]/err_opt)
    title = f"RieADAM with lr={config['lr']} for {config['n_sites']} sites, $t=${config['t']}"
    title += ', '+str(config['n_repetitions']) + ' repetitions'

    plt.figure(dpi=300)
    plt.semilogy(points, err_iter, '.-', label=label)
    plt.xlabel("Iteration")
    plt.ylabel(r"$\mathcal{C}$")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    
    if save:
        fname = 'Le_' + str(config['n_repetitions']) +'_' + str(config['t']) + '_' + config['hamiltonian'] + '_' + str(config['n_iter']) + '_' + str(config['beta_1'])+ '_' + str(config['beta_2']) + '_' + str(config['lr'])+ '_loss.pdf'
        fdir = os.path.join(os.getcwd(), fname)
        plt.savefig(fdir)


def plot_fidelity(df, fidelity, t, epsilon, method, system=None, save_path=None):
    """
    This function plots the fidelity of a given dataset based on the specified fidelity type,
    number of Trotter repetitions, and Hamiltonian type. It returns the elbow and plateau values
    of the fidelity curve.

    Parameters:
        df (pandas.DataFrame): The input dataset as a pandas DataFrame.
        fidelity (str): The type of fidelity to plot.
        t (int): The evolution time for file name.
        system (str): The type of Hamiltonian for the file name.
        save_path (path, optional): The path to save the plot. If not provided, the plot will not be saved.

    Returns:
        tuple: A tuple containing two integers:
            - The elbow value of fidelity: The point in the fidelity curve where the curve bends or "elbows".
            - The plateau value of fidelity: The highest fidelity value reached after the elbow, where the fidelity remains relatively constant.
    """

    N_vals = np.asarray(df['circuit_layers'].values.tolist(), dtype=float)
    fidelities = np.asarray(df[fidelity].values.tolist(), dtype=float)

    # Normalize N and fidelity to [0, 1]
    N_norm = (N_vals - N_vals.min()) / (N_vals.max() - N_vals.min())
    fid_norm = (fidelities - fidelities.min()) / (fidelities.max() - fidelities.min())

    # Line from first to last point
    line_vec = np.array([N_norm[-1] - N_norm[0], fid_norm[-1] - fid_norm[0]])
    line_vec /= np.linalg.norm(line_vec)

    # Compute perpendicular distances to line
    distances = []
    for i in range(len(N_norm)):
        point = np.array([N_norm[i] - N_norm[0], fid_norm[i] - fid_norm[0]])
        projection = np.dot(point, line_vec) * line_vec
        perpendicular = point - projection
        distances.append(np.linalg.norm(perpendicular))

    elbow_idx = int(np.argmax(distances))
    elbow_N = N_vals[elbow_idx]

    plateau_N = None
    FIDELITY_THRESHOLD = fidelities.max() - epsilon

    # Loop through fidelity differences
    for i in range(1, len(fidelities)):
        if fidelities[i] >= FIDELITY_THRESHOLD:
            delta = abs(fidelities[i] - fidelities[i - 1])
            if delta < epsilon:
                if i + 2 < len(fidelities):
                    next_deltas = [abs(fidelities[j] - fidelities[j - 1]) for j in range(i + 1, i + 3)]
                    if all(d < epsilon for d in next_deltas):
                        plateau_N = N_vals[i]
                        break
                else:
                    plateau_N = N_vals[i]
                    break
    
    # Plot
    plt.figure(dpi=300)
    plt.yscale("log")
    plt.plot(N_vals, 1-fidelities, marker='o', label='MPO Infidelity')
    # if elbow_N is not None:
    #     plt.axvline(x=elbow_N, linestyle='--', color='red', label=f'Elbow at N={elbow_N}')
    if plateau_N is not None:
        plt.axvline(x=plateau_N, linestyle='--', color='green', label=f'Threshold infidelity ({epsilon})')
    plt.title(f'Infidelity vs Circuit Depth Ising (with Elbow and Plateau)')
    plt.xlabel('Number of Brickwall Layers (N)')
    plt.ylabel('Infidelity (log sclae)')
    plt.legend()
    plt.grid(True)

    if save_path:
        fname = f"{method}_{t}_{system}_infidelity.pdf"
        fdir = os.path.join(fig_dir, fname)
        plt.savefig(fdir)
    plt.show()

    print(f"\n🔍 Suggested optimal compression depth: N = {plateau_N}")

    return elbow_N, plateau_N


def optimize_generic_circuit_RieADAM(config, U, Vlist_start):
    """
    Optimize the quantum gates in a swap network layout to approximate
    the unitary matrix `U` using a Riemannian ADAM optimizer.
    Vlist_start is given in the form of tensors or matrices.
    """
    
    assert(Vlist_start.shape[1:]==(2,2,2,2))
    f_df = lambda vlist: get_riemannian_gradient_and_cost_function_general(
        U, vlist, max_bondim=config['max_bondim'],
        reference_is_normalized=config['normalize_reference']
    )

    # Define retraction, projection, and inner product
    _retract = lambda v, eta: retract_unitary(v, eta, use_TN=True)
    _project = lambda u,z: project_unitary_tangent(u,z, True)
    retract, project = vmap(_retract), vmap(_project)
    
    # Set up the optimizer
    _metric = lambda v,x,y: inner_product(v,x,y,True)
    metric = vmap(_metric)
    opt = RieADAM(maxiter=config['n_iter'], lr=float(config['lr']))
    Vlist, neval, err_iter = opt.minimize(function=f_df, initial_point=Vlist_start,
                                          retract=retract, projection=project, metric=metric)


    err_iter1 = jnp.asarray(err_iter)
    # Cost function: Frobenius norm
    err_init = err_iter[0]
    err_opt = jnp.min(jnp.asarray(err_iter))
    err_end = err_iter[-1]

    print(f"err_init: {err_init}")
    print(f"err_end after {neval} iterations: {err_end}")
    print(f"err_opt: {err_opt}")
    print(f"err_init/err_opt: {err_init/err_opt}")

    _ = plot_loss(config, err_iter, err_opt, save=True)
                
    return Vlist, err_iter