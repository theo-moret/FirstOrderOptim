# Import 

import numpy as np


# Class

class LinearModel:

    def __init__(self, dim):
        """
        w : coef, array (d,)
        c : intercept, float
        """
        self.dim = dim

        self.params = {
            'coef': np.random.randn(dim), # shape (d,)
            'intercept': np.random.randn(1)
        }

        self.cache = {}


    def forward(self, x):
        """
        Calculate y_pred given x and the current params, then update the cache.
        """
    
        if x.ndim == 1:
            x = x.reshape(-1,self.dim) # shape (N,d)

        y_pred = x @ self.params['coef']  + self.params['intercept'] # shape (N,)

        self.cache['x'] = x 
        self.cache['y_pred'] = y_pred 

        return y_pred 
    
    def backward(self, y_true, loss):
        """
        Return the gradient of the loss with respect to the params.
        """

        x = self.cache['x']
        y_pred = self.cache['y_pred']

        # gradient of the loss wrt y_pred
        dldy = loss.backward(y_pred,y_true) # shape (N,)

        # chain rule
        dldw = np.mean(dldy[:,np.newaxis] * x, axis=0) # shape (d,)
        dldc = np.mean(dldy[:,np.newaxis] * 1, axis=0) # shape (1,)

        return {'coef': dldw, 'intercept': dldc}


    def update(self, new_params):
        """
        Update model params to new_params.
        """
        self.params = new_params
        return None
    
if __name__ == "__main__":
    pass