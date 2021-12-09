import multi_cavity_base
import states
from util import isiter, sort_eigenstates

import numpy as np
import qutip
import copy

class CavityArray(multi_cavity_base.CavityArray):
    def __init__(self, num_cavities, num_photons, model_params, periodic=False):
        """Create new CavityArray object
        Args:
            num_cavities: number of cavities in the array
            num_photons: number of photons
            model_params: dict containing
                'emitters_per_cavity': List or float, number of emitters in each cavity
                'kappa': List or float, cavity decay rate
                'hopping': List or float, cavity-cavity hopping rate
                'gamma': List or float, emitter decay rate
                'g': List or float, cavity emitter coupling constants
                'cavity_freqs': List or float, cavity frequency
                'emitter_freqs': List or float, emitter frequencies
            periodic: bool, True for periodic boundary conditions
        """
        super().__init__(num_cavities, num_photons, model_params, periodic)
        self.states = states.States(self.num_cavities, self.model_params['emitters_per_cavity'], self.num_photons)

    
    def hamiltonian(self):
        H = np.zeros((len(self.states), len(self.states)), dtype='complex')
        
        for col, state in enumerate(self.states):
            for loc in set(state):
                #cavity-emitter interaction terms
                if loc[1] >= 0: #excited emitter
                    #a.dag_i * sigma_ij
                    newstate, n = states.create((loc[0], -1), *states.destroy(loc, state))
                    if newstate in self.states:
                        row = self.states.index(newstate)
                        H[row][col] += self.cavities[loc[0]].g[loc[1]]
                
                #hopping terms
                if loc[1] == -1: #photon in cavity
                    # a.dag_(i+1) * a_i
                    newstate, n = states.create((loc[0]+1, -1), *states.destroy(loc, state))
                    if self.periodic and self.num_cavities > 2: newstate = [(quanta[0]%self.num_cavities, quanta[1]) for quanta in newstate]
                    if newstate in self.states:
                        row = self.states.index(newstate)
                        H[row][col] -= self.hopping[loc[0]] * n
        
        H += H.T #add the transpose terms
        
        #add in a.dag a and sigma.dag sigma terms on the diagonal
        for col, state in enumerate(self.states):
            for loc in set(state):
                newstate, n = states.number(loc, state)
                if loc[1] >= 0: #emitter
                    w = self.cavities[loc[0]].emitter_freqs[loc[1]] - 0.5j * self.cavities[loc[0]].gamma[loc[1]]
                else:
                    w = self.cavities[loc[0]].cavity_freq - 0.5j * self.cavities[loc[0]].kappa
                H[col][col] += w * n
        
        
        return qutip.Qobj(H)

    def eigenstates(self):
        """Wrapper function for numpy.linalg.eig
        Returns the eigenvalues and eigenvectors of the Hamiltonian sorted by energy level
        """
        
        if self._eigenstates is None:
            self._eigenstates = np.linalg.eig(self.hamiltonian())
            self._eigenstates = sort_eigenstates(*self._eigenstates)
        
        return self._eigenstates
