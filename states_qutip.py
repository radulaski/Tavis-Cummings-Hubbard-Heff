import states_base
import multi_cavity_qutip
import numpy as np
import qutip
import copy

class States(states_base.States):
    def __init__(self, cavity_array):
        """Create new States object
        a state is a list of quanta locs given as (cavity, emitter) pairs; 
        emitter == -1 means photons is in the cavity
        Args:
            num_cavities: int number of cavities in the array
            emitters_per_cavity: list of number of emitters in each cavity
            num_photons: int number of photons
        """
        #super().__init__()
        self.states = generate_states(cavity_array)
        self._index = {tuple(state): i for i, state in enumerate(self.states)}
    
    def tovec(self, state):
        vec = np.zeros(len(self), dtype='complex')
        vec[self.index(state)] = 1
        return qutip.Qobj(vec)
        
    

def tovec(cavity_array, state):
    l = []
    for i, cavity in enumerate(cavity_array):
        l.append(qutip.basis(cavity_array.num_photons+1, state.count((i,-1))))
        for j in range(cavity.num_emitters):
            l.append(qutip.basis(2,state.count((i,j))))
    return qutip.tensor(l)
    

def generate_states(cavity_array):

    state_idx = {}
    num_photons = 0
    while len(state_idx) < cavity_array.dim:
        for state in states_base.generate_states(cavity_array.num_cavities, 
                                                 cavity_array.model_params['emitters_per_cavity'],
                                                 num_photons):
            try:
                state_idx[np.nonzero(tovec(cavity_array, state))[0].tolist()[0]] = state
            except:
                pass
        num_photons += 1
    return [state_idx[i] for i in range(cavity_array.dim)]
    
    
    
    
