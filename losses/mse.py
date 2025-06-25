# Import 

import numpy as np

# Class

class MSELoss:

    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        """
        Return the value of the loss at (y_pred,y_true). Return an array of shape (1,).
        """
        return np.mean((y_pred - y_true) ** 2)
    

    def backward(self, y_pred, y_true):
        """
        Return the gradient of the loss wrt to y_pred. Return an array of shape (len(y_pred),).
        """
        return 2 * (y_pred - y_true)/len(y_true) 