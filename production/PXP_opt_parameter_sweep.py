import jax
import yaml
import os
import pickle
from copy import deepcopy

from ropt_aqc.save_model import load_reference
from ropt_aqc.brickwall_circuit import get_initial_gates
from ropt_aqc.tn_helpers import left_to_right_QR_sweep
from ropt_aqc.brickwall_opt import optimize_swap_network_circuit_RieADAM

jax.config.update("jax_enable_x64", True)

def PXP_hyperparameter_sweep(num_sites, num_steps, t, beta_1, beta_2, lr):
    gates_path = f"gates_{num_sites}q_{num_steps}steps_{t}t_{beta_1}_{beta_2}_{lr}.pkl"

    system = 'pxp'
    gates = get_initial_gates(n_sites=num_sites, t=t, n_repetitions=num_steps, degree=1, hamiltonian=system, n_id_layers=0, use_TN=True)

    repo_root = "/home/b/aag/ropt-aqc/"
    config_file = os.path.join(repo_root, "run", system, "configs", "config.yml") 
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    config['reference_dir'] = f"/home/b/aag/ropt-aqc/run/{system}/reference"
    U_ref, t, _, _, _, _, _, _, _ = load_reference(config['reference_dir'], config['n_sites'], config['n_repetitions'], config['t'])
    U_ref = left_to_right_QR_sweep(U_ref, get_norm=False, normalize=config['normalize_reference'])

    config_sweep = deepcopy(config)
    config_sweep['beta_1'] = beta_1
    config_sweep['beta_2'] = beta_2
    config_sweep['lr'] = lr
    results = []

    print(f"\nRunning sweep with beta_1 = {beta_1}, beta_2 = {beta_2}, lr={lr}")
    gates_optimised, err_iter = optimize_swap_network_circuit_RieADAM(config_sweep, U_ref, gates)

    final_err = float(err_iter[-1])
    min_err = float(min(err_iter))

    results.append({
        'beta_1': beta_1,
        'beta_2': beta_2,
        'final_err': final_err,
        'min_err': min_err,
        'lr': lr,
        'err_ratio': float(err_iter[0] / min_err)
    })

    gates_optimised = gates_optimised.reshape((len(gates_optimised), 4, 4))
    with open(gates_path, "wb") as f:
        pickle.dump((gates_optimised, err_iter), f)
