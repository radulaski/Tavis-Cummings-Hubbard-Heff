import multi_cavity_base
import states_qutip
from util import isiter, sort_eigenstates

import numpy as np
import qutip
import copy

class CavityArray(multi_cavity_base.CavityArray):
    def __init__(self, num_cavities, num_photons, model_params, periodic=False):
        super().__init__(num_cavities, num_photons, model_params, periodic)
        
        self.dim = (num_photons+1)**num_cavities*2**np.sum(self.model_params['emitters_per_cavity'])
        self.states = states_qutip.States(self)
    
    def a(self, i):
        """Returns the photon annilihation operator for the ith cavity"""
        l = []
        for c, cavity in enumerate(self):
            if c == i:
                l += [qutip.destroy(self.num_photons+1)]
            else:
                l += [qutip.identity(self.num_photons+1)]
            l += [qutip.identity(2)]*cavity.num_emitters
        return qutip.tensor(l)
        
    def sigma(self, i, j):
        """Returns the emitter lowering operator for the jth emitter in the ith cavity"""
        l = []
        for c, cavity in enumerate(self):
            l += [qutip.identity(self.num_photons+1)]
            if c == i:
                l += [qutip.identity(2)]*j + [qutip.destroy(2)] + [qutip.identity(2)]*(cavity.num_emitters-1-j)
            else:
                l += [qutip.identity(2)]*cavity.num_emitters
        return qutip.tensor(l)
    
    def hamiltonian(self):
        """Returns the full qutip hamiltonian"""
        H = 0
        for i, cavity in enumerate(self):
            a = self.a(i)
            H += cavity.cavity_freq * a.dag() * a
            for j in range(cavity.num_emitters):
                s = self.sigma(i,j)
                H += cavity.emitter_freqs[j] * s.dag() * s + cavity.g[j] * (a.dag()*s + s.dag() * a)
        
        #hopping terms
        a = self.a(0)
        for i, J in enumerate(self.hopping):
            if self.periodic and self.num_cavities > 2 and i == len(self.hopping):
                a1 = self.a(0)
            else:
                a1 = self.a(i+1)
            H -= J * (a.dag()*a1 + a1.dag()*a)
            a = a1
        return H

    def eigenstates(self):
        """Wrapper function for Qobj.eigenstates()
        Args:
        Returns the eigenvalues and eigenvectors of the Hamiltonian sorted by energy level
        """
        if self._eigenstates is None:
            self._eigenstates = self.hamiltonian().eigenstates()
        return self._eigenstates
