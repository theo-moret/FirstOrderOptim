# Import 

import numpy as np
from abc import ABC, abstractmethod

# Class

class BaseOptimizer(ABC):
    """
    Base for all optimizer objects.

    Subclasses must implement 'step'.
    """

    def __init__(self, learning_rate: float):
        self.lr = learning_rate


    @abstractmethod
    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ 
        For the dict of params dict[str, np.ndarray] with current values and associated grads dict[str, np.ndarray] 
        derived from backward methods, implement the optimizer step.

        Args:
        - params (dict[str, np.ndarray]): dict with params as keys and associated current values.
        - grads (dict[str, np.ndarray]): dict with params as keys and associated values of the derivative of the loss wrt the params.

        Return:
        - (dict[str, np.ndarray]): dict of new params to update the model.
        """


if __name__ == "__main__":
    pass