import numpy as np
from typing import List, Tuple, Dict, Union

import multi_cavity
import states
from util import *

def participation(cavity_array, normalize=False):
    """
    Args:
        cavity_array: multi_cavity.CavityArray object
    Returns:
        the participation ratio, p, for the eigenstates of cavity_array
        p = 1/sum_i abs(v_i)^4 where v_i are the eigenvector components
    """
    
    eig_vals, eig_vecs = cavity_array.eigenstates()
    p = 1/(np.sum(abs(eig_vecs)**4, axis=0))
    if normalize:
        p = (p-1)/(len(eig_vals)-1)
    return p

def node_participation(cavity_array, normalize=False):
    """
    Only for single photon cavity array
    Args:
        cavity_array: multi_cavity.CavityArray object
    Returns:
    """
    
    eig_vals, eig_vecs = cavity_array.eigenstates()
    p = np.zeros((cavity_array.num_cavities, eig_vecs.shape[-1]))
    for state, v_i in zip(cavity_array.states, eig_vecs):
        cav = state[0][0]
        p[cav] += abs(v_i)**2
    p = 1/np.sum(p**2, axis=0)
    if normalize:
        p = (p-1)/(cavity_array.num_cavities-1)
    return p

def polariton_participation(cavity_array, normalize=False):
    """
    Only for single photon cavity array
    Args:
        cavity_array: multi_cavity.CavityArray object
    Returns:
        participation ratio for cavity or emitter components
    """
    
    eig_vals, eig_vecs = cavity_array.eigenstates()
    p = np.zeros((2, eig_vecs.shape[-1]))
    for state, v_i in zip(cavity_array.states, eig_vecs):
        cav = state[0][0]
        if state[0][1] == -1:
            p[0] += abs(v_i)**2
        else:
            p[1] += abs(v_i)**2
    p = 1/np.sum(p**2, axis=0)
    if normalize:
        p = p-1
    return p


def photon_expect(cavity_array):
    """
    Args:
        cavity_array: multi_cavity.CavityArray object
    Returns:
        photon expectation value for each cavity for each eigenstate
    """
    eig_vals, eig_vecs = cavity_array.eigenstates()
    expect_vals = np.zeros(( eig_vecs.shape[-1], cavity_array.num_cavities))
    
    for i in range(cavity_array.num_cavities):
        cavity_exp = 0
        for k, row in enumerate(eig_vecs):
            _, n = states.number((i,-1), cavity_array.states[k])
            cavity_exp += n * abs(row)**2
        expect_vals[:,i] = cavity_exp
    return expect_vals

def excite_expect(cavity_array):
    """
    Args:
        cavity_array: multi_cavity.CavityArray object
    Returns:
        sum of emitter excitation expectation values at each cavity for each eigenstate
    """
    eig_vals, eig_vecs = cavity_array.eigenstates()
    expect_vals = np.zeros(( eig_vecs.shape[-1], cavity_array.num_cavities))
    
    for i, cavity in enumerate(cavity_array):
        cavity_exp = 0
        for j in range(cavity.num_emitters):
            for k, row in enumerate(eig_vecs):
                _, n = states.number((i,j), cavity_array.states[k])
                cavity_exp += n * abs(row)**2
        expect_vals[:,i] = cavity_exp
    return expect_vals

def expect(cavity_array, locs):
    """
    Args:
        cavity_array: multi_cavity.CavityArray object
        locs: list[(cavity, emitter)] of which locs to include
    Returns:
        sum of expectation values at each cavity loc for each eigenstate
    """
    eig_vals, eig_vecs = cavity_array.eigenstates()
    expect_vals = np.zeros(eig_vecs.shape[-1])

    for loc in locs:
        N = np.zeros(len(cavity_array.states))
        for i, state in enumerate(cavity_array.states):
            _, n = states.number(loc, state)
            N[i] = n
        expect_vals += np.abs(eig_vecs.T)**2 @ N
    return expect_vals



def number(cavity_array, loc, vecs):
    """
    Args:
        cavity_array
        loc: (cavity, emitter) tuple
        vecs: row vectors the number operator is acting on
    Returns:
        row vectors of N_loc acting on vecs
    """
    N = np.zeros(len(cavity_array.states))
    for i, state in enumerate(cavity_array.states):
        _, n = states.number(loc, state)
        N[i] = n
    return vecs * N

