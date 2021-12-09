import single_cavity
from util import isiter

import numpy as np
from collections import Sequence
import copy

class CavityArray(Sequence):
    def __init__(self, num_cavities, num_photons, model_params, periodic=False):
        """Create new CavityArray object as a sequence of single_cavity objects
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
    
        #set object attributes
        self.num_cavities = num_cavities
        self.num_photons = num_photons
        self.periodic = periodic
    
        self.model_params = setup_model_params(self.num_cavities, model_params, self.periodic)
        self.cavities = setup_cavities(self.num_cavities, self.model_params)
        self.states = None
        
        #caching eigenstates
        self._eigenstates = None
        
    #sequence class methods
    def __len__(self): return self.num_cavities
    def __getitem__(self, i): return self.cavities[i]

    @property
    def hopping(self): 
        """Returns the cavity-cavity hopping rate"""
        return self.model_params['hopping']
    
    def hamiltonian(self):
        pass

    def eigenstates(self):
        pass

#CavityArray setup functions

def setup_model_params(num_cavities, input_model_params, periodic):
    """Helper function to setup and validate input model parameters
    """
    keys = ['emitters_per_cavity', #start with emitters per cavity
            'kappa',
            'hopping',
            'gamma',
            'emitter_freqs',
            'cavity_freqs',
            'g']
    emitter_keys = ['gamma', 'g', 'emitter_freqs']
    model_params = dict.fromkeys(keys)
    for key in keys:
        vals = copy.deepcopy(input_model_params[key])
        expected_length = num_cavities #length  of list
        if key == 'hopping' and (not periodic or num_cavities <= 2):
            expected_length -= 1
        expected_length = max(0, expected_length)
        
        if not isiter(vals): #then convert to list
            vals = [vals] * expected_length
        
        #validate len of list
        assert len(vals) == expected_length, "The '{}' list should have length {}, got {}".format(key, expected_length, len(vals))
        
        if key in emitter_keys:
            for i in range(num_cavities):
                expected_length = model_params['emitters_per_cavity'][i]
                if not isiter(vals[i]):
                    vals[i] = [vals[i]] * expected_length
                
                assert len(vals[i]) == expected_length, "The {}th cavity '{}' list should have length {}, got {}".format(i, key, expected_length, len(vals[i]))
        model_params[key] = vals
    return model_params
    

def setup_cavities(num_cavities, model_params):
    """Helper function to setup list of single cavities
    """
    cavities = []
    for i in range(num_cavities):
        cavity_model_params = {
            'num_emitters' : model_params['emitters_per_cavity'][i],
            'kappa': model_params['kappa'][i],
            'gamma': model_params['gamma'][i],
            'cavity_freq': model_params['cavity_freqs'][i],
            'emitter_freqs': model_params['emitter_freqs'][i],
            'g': model_params['g'][i],
            }
        
        cavity = single_cavity.Cavity(cavity_model_params)
        cavities.append(cavity)
    return cavities
