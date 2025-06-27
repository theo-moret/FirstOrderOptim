# Import 

import numpy as np

from first_order_optim.loss.base import BaseLoss
from first_order_optim.model.base import BaseModel

# Class

class LinearModel(BaseModel):

    def __init__(self, dim: int):
        """
        Args:
        - dim (int): dimension of the model, ie. number of scalar covariates.
        """
        super().__init__()

        self.dim = dim

        self.params['coef'] = np.random.randn(dim) # shape (d,)
        self.params['intercept'] = np.random.randn(1) # shape (1,)

        self.cache = {}


    def forward(self, x: np.ndarray, override_params: dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Return y_pred given x and the current params, then update the cache.

        - d : dimension of parameters coef
        - N : number of observations in x, ie. batch_size

        Args:
        - x (np.ndarray of shape (N,d) or (N,) if d=1): input data.
        - override_params (dict[str, np.ndarray] or None): If provided, use these parameters instead of self.params, to compute look-ahead gradient, eg. Nesterov Momentum. 

        Return:
        - (np.ndarray of shape (N,))
        """

        # Reshape x if d=1 to perform next matrix multiplication
        if x.ndim == 1:
            x = x.reshape(-1,self.dim) # shape (N,d)

        y_pred = x @ self.params['coef']  + self.params['intercept'] # shape (N,)

        # Update the cache only if we are not override_params, ie not for look-ahead.
        if override_params is None:
            self.cache['x'] = x 
            self.cache['y_pred'] = y_pred 

        return y_pred 
    


    def backward(self, y_true: np.ndarray, loss: BaseLoss, override_params: dict[str, np.ndarray] = None) -> dict[str, np.ndarray]:
        """
        Return the gradient of the loss with respect to the params. 

        Args:
        - y_true (np.ndarray): true label from the data.
        - loss (BaseLoss): loss object from .loss
        - override_params (dict[str, np.ndarray] or None): If provided, use these parameters instead of self.params, to compute look-ahead gradient, eg. Nesterov Momentum.

        Return:
        -(dict[str, np.ndarray]) containing the value of the gradient of the loss with respect to each params.
        """
            
        x = self.cache['x'] # shape (N,d)

        # To handle look-ahead
        if override_params is not None:
            y_pred = x @ override_params['coef'] + override_params['intercept']
        else:
            y_pred = self.cache['y_pred']

        # Gradient of the loss wrt y_pred
        dldy = loss.backward(y_pred,y_true) # shape (N,)

        # Chain rule, already divided by 1/N in dldy, no need to np.mean
        dldw = (dldy[:,np.newaxis] * x).sum(axis=0) # shape (d,)
        dldc = dldy.sum() # shape (1,)

        return {'coef': dldw, 'intercept': dldc}

    
if __name__ == "__main__":
    pass