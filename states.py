import states_base
import numpy as np
import copy

class States(states_base.States):
    def __init__(self, num_cavities, emitters_per_cavity, num_photons):
        """Create new States object
        a state is a list of quanta locs given as (cavity, emitter) pairs; 
        emitter == -1 means photons is in the cavity
        Args:
            num_cavities: int number of cavities in the array
            emitters_per_cavity: list of number of emitters in each cavity
            num_photons: int number of photons
        """
        #super().__init__()
        self.states = states_base.generate_states(num_cavities, emitters_per_cavity, num_photons)
        self._index = {tuple(state): i for i, state in enumerate(self.states)}
    
    def tovec(self, state):
        vec = np.zeros(len(self), dtype='complex')
        vec[self.index(state)] = 1
        return vec
    

#operators that act on the basis states
#[] is equivalent to the zero vector
def number(loc, state, multiplier=1):
    """number operator; equivalent to a.dag a  or sigma.dag sigma
    Args:
        loc: (cavity, emitter) tuple
        state: basis state given as list of quants locs
        multiplier: scaler
    Returns (state, val * multiplier) where val is the number of quanta at
    loc = (cavity, emitter)"""
    
    n = state.count(loc) #len([1 for quanta in state if quanta == loc])
    if n == 0 or multiplier == 0:
        return ([], 0)
    return (state, n * multiplier)
    
def destroy(loc, state, multiplier=1):
    _, n = number(loc, state)
    if n == 0 or multiplier == 0:
        return ([], 0)
    newstate = copy.deepcopy(state)
    newstate.pop(newstate.index(loc))
    return (newstate, np.sqrt(n) * multiplier)

def create(loc, state, multiplier=1):
    if loc[1] >= 0: #excited emitters cannot be raised
        return ([], 0)
    _, n = number(loc, state)
    if multiplier == 0:
        return ([], 0)
    newstate = copy.deepcopy(state)
    newstate += [loc]
    newstate.sort()
    return (newstate, np.sqrt(n+1) * multiplier)

