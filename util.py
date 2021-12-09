import numpy as np
import inspect
from typing import List, Tuple, Dict, Union



def isiter(obj) -> bool:
    """Returns true if obj is an iterable object; false otherwise"""
    try:
        iter(obj)
        return True
    except TypeError:
        return False
    

def sort_eigenstates(eig_vals: np.ndarray, eig_vecs: np.ndarray, sortby: str='energy') -> Tuple[np.ndarray, np.ndarray]:
    pairs = list(zip(eig_vals, eig_vecs.T))
    if sortby == 'energy':
        pairs.sort(key=lambda pair: pair[0].real)
    elif sortby == 'participation':
        pairs.sort(key=lambda pair: 1/(np.sum(abs(pair[1])**4)))
    else:
        print("Accecpted sortby values are 'energy', 'participation', cannot sort by {}".format(sortby))
        return eig_vals, eig_vecs
    _eig_vals, _eig_vecs = zip(*pairs)
    return np.array(_eig_vals), np.array(_eig_vecs).T



def kwargs_sep(fcn, kwargs):
    """Used to separate kwargs for multiple different functions
    Args:
        fcn: function
        kwargs: dict of keyword args 
    Returns:
        dict for fcn keywords contained in kwargs"""
    
    #list of fcn argument names
    fcn_args = [key for key, val in inspect.signature(fcn).parameters.items()]
    #dict of kwargs for fcn
    fcn_kwargs = {key: kwargs[key] for key in kwargs if key in fcn_args}
    return fcn_kwargs
