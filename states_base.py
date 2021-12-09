import numpy as np
import copy
from collections import Sequence
from itertools import combinations_with_replacement

class States(Sequence):
    def __init__(self):
        """Create new States object
        Args:
            num_cavities: int number of cavities in the array
            emitters_per_cavity: list of number of emitters in each cavity
            num_photons: int number of photons
        """
        self.states = []
        self._index = {}
    
    #sequence class methods
    def __len__(self): return len(self.states)
    def __getitem__(self, i): return self.states[i]
    def __contains__(self, state): 
        state.sort()
        return tuple(state) in self._index
    
    def index(self, state):
        """Returns the index of the state"""
        state.sort()
        return self._index[tuple(state)]
    
    def tovec(self, state):
        pass
    
    def labels(self):
        return labels(self.states)
    

def generate_states(num_cavities, emitters_per_cavity, num_photons):
    if num_photons == 0: return [[]]
    states = []
    cavity_dist = list(combinations_with_replacement(range(num_cavities), num_photons))
    for cavities in cavity_dist:
        args = [range(-1, emitters_per_cavity[cav]) for cav in cavities]
        grid = np.array(np.meshgrid(*args)).T.reshape(-1, num_photons)
        grid = np.unique(grid, axis=0)
        grid = [list(zip(cavities, row)) for row in grid]
        for row in grid: row.sort()
        grid = np.unique(grid, axis=0).tolist()
        for row in grid:
            addrow = True
            for quanta in row:
                if quanta[1] != -1:
                    if row.count(quanta) > 1:
                        addrow = False
                        break
            if addrow:
                states.append([tuple(quanta) for quanta in row])
    return states

def labels(states):
    """
    Args:
        states: list of states
    Returns:
        list of labels for the vector components in the same order as the hamiltonian
        in Latex for plots; indexed from 0
        ex. e0,1 is emitter 1 in cavity 0
    """
    labels = []
    for state in states:
        s = list(set(state)) #unique locations only
        s.sort()
        label = ''
        for quanta in s:
            if quanta[1] == -1:
                m = state.count(quanta)
                m = str(m) if m>1 else ''
                label += str(m) + '$c_{' + '{}'.format(quanta[0]) + '}$'
            else:
                label += '$e_{'+'{},{}'.format(*quanta) + '}$'
        labels.append(label)
    return labels
