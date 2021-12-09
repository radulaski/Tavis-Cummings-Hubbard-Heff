"""Class used for multi_cavity.CavityArray class to create sequence of single cavities"""
from typing import List, Tuple, Dict, Union

class Cavity:
    def __init__(self, model_params) -> None:
        """
        Creates a new Cavity object
        Args:
            model_params: dict containing
                'num_emitters': number of emitters in the cavity
                'kappa': float, cavity decay rate
                'gamma': List or float, emitter decay rate
                'g': List or float, cavity emitter coupling constants
                'cavity_freq': float, cavity frequency
                'emitter_freqs': List or float, emitter frequencies
        """
        
        #set object attributes
        self._model_params = model_params
        
    @property
    def num_emitters(self) -> int: 
        """Returns the number of emitters in the cavity"""
        return self._model_params['num_emitters']

    @property
    def kappa(self) -> float: 
        """Returns the cavity decay rate"""
        return self._model_params['kappa']

    @property
    def gamma(self) -> List[float]: 
        """Returns decay rate of the emitters"""
        return self._model_params['gamma']

    @property
    def emitter_freqs(self) -> List[float]: 
        """Returns the emitter frequenices"""
        return self._model_params['emitter_freqs']
        
    @property
    def g(self) -> List[float]: 
        """Returns the cavity emitter coupling constants"""
        return self._model_params['g']

    @property
    def cavity_freq(self) -> float: 
        """Returns the cavity frequency"""
        return self._model_params['cavity_freq']







