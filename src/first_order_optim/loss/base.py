# Import 

import numpy as np
from abc import ABC, abstractmethod


# Class

class BaseLoss(ABC):
    """
    Base for all loss objects.

    Subclasses must implement 'forward' and 'backward'.
    """

    def __init__(self):
        self.params: dict[str, np.ndarray | float] = {}


    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Return the value of the loss at (y_pred,y_true) as an array of shape (1,).

        Args:
        - y_pred (np.ndarray of shape (N,..)): output predicted by the model for N samples.
        - y_true (np.ndarray of shape (N,..)): true label from the data for N samples. 

        Return:
        - (np.ndarray of shape (1,)) 
        """


    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Return the gradient of the loss wrt to y_pred as an array of shape (N,).

        Args:
        - y_pred (np.ndarray of shape (N,..)): output predicted by the model for N samples.
        - y_true (np.ndarray of shape (N,..)): true label from the data for N samples.  

        Return:
        - (np.ndarray of shape (N,)) 
        """

    