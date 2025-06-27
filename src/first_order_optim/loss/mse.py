# Import 

import numpy as np

from first_order_optim.loss.base import BaseLoss


# Class

class MSELoss(BaseLoss):

    def __init__(self):
        pass

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        See BaseLoss.forward, but return the value of the loss at (y_pred,y_true).
        """
        return np.mean((y_pred - y_true) ** 2)
    

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        See BaseLoss.backward, but return the gradient of the loss wrt to y_pred.
        """
        return 2 * (y_pred - y_true)/len(y_true) 