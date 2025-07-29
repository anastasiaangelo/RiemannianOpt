# From Isabel's code

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time
import yaml

from time import time

from jax.numpy import asarray
from jax import config
config.update("jax_enable_x64", True)

from ropt_aqc.save_model import save_reference
from ropt_aqc.initialise_hamiltonians import get_hamiltonian_terms
from ropt_aqc.fermionic_systems import construct_spinful_FH1D_hamiltonian
from ropt_aqc.tn_helpers import (get_id_mpo, convert_mpo_to_mps, get_maximum_bond_dimension, 
                                   get_left_canonical_mps, inner_product_mps, compress_mpo)
from ropt_aqc.tn_brickwall_methods import contract_layers_of_swap_network_with_mpo
from ropt_aqc.brickwall_circuit import get_gates_per_layer, get_initial_gates

from ropt_aqc.pxp_model import pxp_hamiltonian_sparse

from helpers import get_duration

def compute_error_mpo(mpo1, mpo2):
    '''
    Compute the Frobenius norm and Hilbert-Schmidt test.
    '''
    # Convert MPOs to MPSs
    mps1, mps2 = convert_mpo_to_mps(mpo1), convert_mpo_to_mps(mpo2)
    # Normalize the MPS
    mps1_nrmd = get_left_canonical_mps(mps1, normalize=True, get_norm=False)
    mps2_nrmd = get_left_canonical_mps(mps2, normalize=True, get_norm=False)
    # Compute overlap
    tr = inner_product_mps(mps1_nrmd, mps2_nrmd)
    err = 2 - 2*tr.real  # Frobenius norm
    return err

def compute_maximum_duration(config, **kwargs):
    t = config['t_start']
    bond_dim = config['max_bond_dim']
    mpo_id = get_id_mpo(config['n_sites'])
    err = 0.
    while err<=1e-10:
        print('Current time: ', t)
        gates = get_initial_gates(config['n_sites'], t, config['n_repetitions'], config['degree'], 
                                  hamiltonian=config['hamiltonian'], use_TN=True, **kwargs)
        gates_per_layer, gate_sites_per_layer, layer_is_odd = get_gates_per_layer(
            gates, config['n_sites'], config["degree"], config["n_repetitions"], hamiltonian=config['hamiltonian'])
        mpo_init = contract_layers_of_swap_network_with_mpo(mpo_id, gates_per_layer, layer_is_odd, gate_sites_per_layer, layer_is_left=True, 
                                                            max_bondim=bond_dim, get_norm=False)
        mpo_reduced = contract_layers_of_swap_network_with_mpo(mpo_id, gates_per_layer, layer_is_odd, gate_sites_per_layer, layer_is_left=True, 
                                                               max_bondim=bond_dim-1, get_norm=False)
        err = compute_error_mpo(mpo_reduced, mpo_init)
        t += 0.1
    t_end = t - 0.1
    print('Maximum simulation time: ', t_end)
    
def compute_reference(config, ref_batch, ref_seed, **kwargs):
    tstart = time()

    # We start with an initial MPO with maximum bond dimension
    mpo_id = get_id_mpo(config['n_sites'])    
    gates = get_initial_gates(config['n_sites'], config['t'], config['n_repetitions'], config['degree'], 
                              hamiltonian=config['hamiltonian'], use_TN=True, **kwargs
                              )
    print('number of agtes:', gates.shape)
    gates_per_layer, gate_sites_per_layer, layer_is_odd = get_gates_per_layer(
        gates, config['n_sites'], config["degree"], config["n_repetitions"], hamiltonian=config['hamiltonian'])
    # print(f"Layers: {len(gates_per_layer)}, Sites per layer: {[len(s) for s in gate_sites_per_layer]}")
    # for l, sites in enumerate(gate_sites_per_layer):
    #     for (s1, s2) in sites:
    #         if layer_is_odd[l] and s1 % 2 != 0:
    #             print(f"[Layer {l}] ❌ Odd layer gate on non-even site: ({s1},{s2})")
    #         elif not layer_is_odd[l] and s1 % 2 != 1:
    #             print(f"[Layer {l}] ❌ Even layer gate on non-odd site: ({s1},{s2})")
    # Check if gates_per_layer[l] and gate_sites_per_layer[l] match in length
    # for l in range(len(gates_per_layer)):
    #     assert len(gates_per_layer[l]) == len(gate_sites_per_layer[l]), f"Layer {l} mismatch"

    # Obtain MPO representation
    bond_dim = config['max_bond_dim']
    mpo_init = contract_layers_of_swap_network_with_mpo(
        mpo_id, gates_per_layer, layer_is_odd, gate_sites_per_layer, layer_is_left=True, max_bondim=bond_dim, get_norm=False)
    assert len(mpo_init) == config['n_sites'], "MPO length mismatch"
    for i, t in enumerate(mpo_init):
        assert t.shape[1] == 2 and t.shape[2] == 2, f"Unexpected physical dims at site {i}: {t.shape}"
    compress = config.get('compress', True)
    if compress:
        degree_thres = config.get('degree_thres', 2)
        n_rep_thres = config.get('n_rep_thres', 10)
        gates_thres = get_initial_gates(
            config['n_sites'], config['t'], n_rep_thres, degree=degree_thres, hamiltonian=config['hamiltonian'], use_TN=True, **kwargs)
        print('number of agtes thresh:', gates.shape)
        gates_per_layer_thres, gate_sites_per_layer_thresh, layer_is_odd_thres = get_gates_per_layer(
            gates_thres, config['n_sites'], degree=degree_thres, n_repetitions=n_rep_thres, hamiltonian=config['hamiltonian'])
        # print(f"Layers thresh: {len(gates_per_layer_thres)}, Sites per layer: {[len(s) for s in gate_sites_per_layer_thresh]}")
        # Obtain MPO representation
        mpo_thres = contract_layers_of_swap_network_with_mpo(
            mpo_id, gates_per_layer_thres, layer_is_odd_thres, gate_sites_per_layer_thresh, layer_is_left=True, max_bondim=config['max_bond_dim'], get_norm=False)
        err_threshold = compute_error_mpo(mpo_thres, mpo_init)
        fac_thres = config.get('fac_thres', 500)
        err_threshold = err_threshold/fac_thres
        print("\t Error threshold = ", err_threshold)
        
        print("Initial MPO shape:", [t.shape for t in mpo_id])  # before gate application

        print("MPO after applying gates:", [t.shape for t in mpo_init])

        # Compress the initial MPO down to convergence criteria
        step_size = 1
        bond_dim_comp = bond_dim
        err2 = 0.
        while err2<err_threshold:
            if bond_dim_comp <= 2:
                print("⚠️ Reached minimum bond dimension. Aborting compression.")
                break
            bond_dim_comp -= step_size
            mpo = compress_mpo(mpo_init, bond_dim_comp)
            err2 = compute_error_mpo(mpo_init, mpo)
            print(f"\t Errors for bond dims {bond_dim_comp}: {err2}")

        # Get information about final reference
        print('Final MPO has maximum bond dimension: ', bond_dim_comp)
        print('\nFinal error between reference MPO and converged MPO: ', err2)
    else:
        print('Final MPO has maximum bond dimension: ', get_maximum_bond_dimension(mpo_init))
        mpo = mpo_init
        err_threshold = None
    
    # Save the reference
    path = os.path.join(os.getcwd(), config['hamiltonian'], "reference")
    print('saving to path:' , path)

    _ = save_reference(path, mpo, config["t"], config["n_sites"], config["degree"], config["n_repetitions"], 
                       err_threshold=err_threshold, hamiltonian=config['hamiltonian'], H=None, ref_seed=ref_seed, ref_nbr=ref_batch, **kwargs)
    print('saved to path:' , path)
    get_duration(tstart, program='cycle')
    print('\n\n')

    # compress = config.get('compress', True)
    # if compress and config.get('degree_thres', 2) < config['degree']:
    #     degree_thres = config.get('degree_thres', 2)
    #     n_rep_thres = config.get('n_rep_thres', 10)
    #     gates_thres = get_initial_gates(
    #         config['n_sites'], config['t'], n_rep_thres, degree=degree_thres, hamiltonian=config['hamiltonian'], use_TN=True, **kwargs)
    #     gates_per_layer_thres, layer_is_odd_thres = get_gates_per_layer(
    #         gates_thres, config['n_sites'], degree=degree_thres, n_repetitions=n_rep_thres, hamiltonian=config['hamiltonian'])
    #     mpo_thres = contract_layers_of_swap_network_with_mpo(
    #         mpo_id, gates_per_layer_thres, layer_is_odd_thres, layer_is_left=True, max_bondim=config['max_bond_dim'], get_norm=False)

    #     err_threshold = compute_error_mpo(mpo_thres, mpo_init)
    #     fac_thres = config.get('fac_thres', 500)
    #     err_threshold = err_threshold/fac_thres
    #     print("\t Error threshold = ", err_threshold)

    #     if err_threshold <= 0:
    #         print("\t Warning: Negative error threshold, skipping compression.")
    #         mpo = mpo_init
    #     else:
    #         # Compress the initial MPO down to convergence criteria
    #         step_size = 1
    #         bond_dim_comp = bond_dim
    #         err2 = 0.
    #         while err2 < err_threshold:
    #             bond_dim_comp -= step_size
    #             mpo = compress_mpo(mpo_init, bond_dim_comp)
    #             err2 = compute_error_mpo(mpo_init, mpo)
    #             print(f"\t Errors for bond dims {bond_dim_comp}: {err2}")

    #         # Get information about final reference
    #         print('Final MPO has maximum bond dimension: ', bond_dim_comp)
    #         print('\nFinal error between reference MPO and converged MPO: ', err2)

    # else:
    #     print('Final MPO has maximum bond dimension: ', get_maximum_bond_dimension(mpo_init))
    #     mpo = mpo_init
    #     err_threshold = None

    
    # Save the reference
    path = os.path.join(os.getcwd(), config['hamiltonian'], "reference")
    print('saving to path:' , path)

    _ = save_reference(path, mpo, config["t"], config["n_sites"], config["degree"], config["n_repetitions"], 
                       err_threshold=err_threshold, hamiltonian=config['hamiltonian'], H=None, ref_seed=ref_seed, ref_nbr=ref_batch, **kwargs)
    print('saved to path:' , path)
    get_duration(tstart, program='cycle')
    print('\n\n')


def main():
    t0 = time()
    # Read in config
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    # Set modus
    if 'compute_maximum_time' not in config.keys(): config['compute_maximum_time']=False
    if 'compute_reference' not in config.keys(): config['compute_reference']=False
    if config['hamiltonian']=='fermi-hubbard-1d' and 'n_sites' not in config.keys(): config['n_sites']=2*config['n_orbitals']

    # Set the reference number
    lim = config.get('reference_number_start', 1)
    ref_batches = range(lim, lim+len(config["reference_seed"]))

    disordered='disordered' if config['disordered'] else 'non-disordered'

    # Compute the reference
    print('Start computing MPO reference ({}) for {} with {} sites...'.format(
        disordered, config['hamiltonian'], config['n_sites']))
    if config['hamiltonian']=='fermi-hubbard-1d':
        for ref_batch, ref_seed in zip(ref_batches, config["reference_seed"]):
            _, T, V = construct_spinful_FH1D_hamiltonian(
                config['n_orbitals'], get_matrix=False, disordered=config['disordered'], reference_seed=ref_seed)    
            if config['compute_maximum_time']: compute_maximum_duration(config, T=-T, V=-V)
            if config['compute_reference']: compute_reference(config, ref_batch, ref_seed, T=-T, V=-V)

    elif config['hamiltonian']=='ising-1d':
        for ref_batch, ref_seed in zip(ref_batches, config["reference_seed"]):
            J, g, h = -config['J'], -config['g'], -config['h']
            _, params = get_hamiltonian_terms(config['n_sites'], system='ising-1d', J=J, g=g, h=h)
            Js = params['J']
            hs = params['h']
            gs = params['g']
            if config['compute_maximum_time']: compute_maximum_duration(config, J=Js, g=gs, h=hs)
            if config['compute_reference']: compute_reference(config, ref_batch, ref_seed, J=Js, g=gs, h=hs)

    elif config['hamiltonian']=='heisenberg':
        for ref_batch, ref_seed in zip(ref_batches, config["reference_seed"]):
            J, h = -asarray(config['J']), -asarray(config['h'])
            _, params = get_hamiltonian_terms(config['n_sites'], system='heisenberg', J=J, h=h)
            Js = params['J']
            hs = params['h']
            if config['compute_maximum_time']: compute_maximum_duration(config, J=Js, h=hs)
            if config['compute_reference']: compute_reference(config, ref_batch, ref_seed, J=Js, h=hs)

    elif config['hamiltonian']=='custom':
        for ref_batch, ref_seed in zip(ref_batches, config["reference_seed"]):
            h_terms, params = get_hamiltonian_terms(config['n_sites'], system='custom', pauli_terms=config['terms'])
            if config['compute_maximum_time']: compute_maximum_duration(config, hamiltonian_terms=params)
            if config['compute_reference']: compute_reference(config, ref_batch, ref_seed, hamiltonian_terms=params)

    elif config['hamiltonian']=='pxp':
        for ref_batch, ref_seed in zip(ref_batches, config["reference_seed"]):
            pxp_ham = pxp_hamiltonian_sparse(config['n_sites'])
            # if config['compute_maximum_time']: compute_maximum_duration(config, hamiltonian_terms=params)
            if config['compute_reference']: compute_reference(config, ref_batch, ref_seed, hamiltonian_terms=pxp_ham)

    get_duration(t0)

if __name__ == "__main__":
    main()