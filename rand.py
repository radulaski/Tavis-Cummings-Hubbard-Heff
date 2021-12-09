"""Functions to generate random paramerter values for multi_cavity.CavityArray"""

import numpy as np
from typing import List, Tuple, Dict, Union
from util import isiter


def cavity_freqs(num_cavities: int, center: float, spread: float) -> List[float]:
    """Wrapper for numpy normal distribution to generate random cavity frequencies
    Args:
        num_cavities: number of num_cavities
        center: center of normal distribution; list or float
        spread: FWHM of the distribution; list or float
    Returns random emitter frequencies
    """
    
    return np.random.normal(center, spread/2, num_cavities).tolist()

def emitters_per_cavity(num_cavities: int, low: int, high: int) -> List[int]:
    """Wrapper for numpy randint to generate random number of emitters per cavity
    Args:
        num_cavities: number of num_cavities
        low: lowest to be drawn from
        high: higest to be drawn from
    Returns random emitter frequencies
    """
    
    return np.random.randint(low, high+1, num_cavities).tolist()

def emitter_freqs(num_cavities: int, emitters_per_cavity: Union[int, List[int]], center: Union[List[float], float], spread: Union[List[float], float]) -> List[List[float]]:
    """Wrapper for numpy normal distribution to generate random emitter frequencies
    Args:
        num_cavities: number of num_cavities
        emitters_per_cavity: number of emitters per cavity; list or int
        center: center of normal distribution; list or float
        spread: FWHM of the distribution; list or float
    Returns random emitter frequencies
    """
    
    #convert to lists for each cavity
    if not isiter(emitters_per_cavity): emitters_per_cavity = [emitters_per_cavity]*num_cavities
    if not isiter(center): center = [center] * num_cavities
    if not isiter(spread): spread = [spread] * num_cavities
    return [np.random.normal(c, s/2, N).tolist() for c, s, N in zip(center, spread, emitters_per_cavity)]


def g(num_cavities: int, emitters_per_cavity: Union[int, List[int]], gmin: Union[List[float], float], gmax: Union[List[float], float]) -> List[List[float]]:
    """Wrapper for numpy uniform distribution to generate random cavity emitter coupling constants
    Args:
        num_cavities: number of num_cavities
        emitters_per_cavity: number of emitters per cavity; list or int
        gmin: min cavity emitter coupling constant
        gmax: max cavity emitter coupling constant
    Returns random cavity emitter coupling constants
    """
    
    #convert to lists for each cavity
    if not isiter(emitters_per_cavity): emitters_per_cavity = [emitters_per_cavity]*num_cavities
    if not isiter(gmin): gmin = [gmin] * num_cavities
    if not isiter(gmax): gmax = [gmax] * num_cavities
    return [(np.random.random(N)*(gmx-gmn)+gmn).tolist() for gmx, gmn, N in zip(gmax, gmin, emitters_per_cavity)]


