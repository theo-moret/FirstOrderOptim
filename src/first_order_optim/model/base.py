# Import 

import numpy as np 
from abc import ABC, abstractmethod


# Class 

class BaseModel(ABC):
    """
    Base for all model objects.

    Subclasses must implement 'forward' and 'backward'.
    """

    def __init__(self):
        self.params: dict[str, np.ndarray] = {}


    @abstractmethod
    def forward(self, *args, **kwargs) -> np.ndarray | float:
        """ Compute outputs of the model given inputs. """


    @abstractmethod
    def backward(self, *args, **kwars) -> dict[str, np.ndarray | float]:
        """ Return a dictionary with params as keys and the associated value of the gradient of the loss wrt the params."""

    def update(self, new_params):
        """
        Update model params to new_params.

        Args:
        - new_params (dict[str, np.ndarray | float]): dict of params and associated values from which the model will be updated.
        """
        self.params = new_params
        return None