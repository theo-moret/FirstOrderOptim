# Import 

import numpy as np


# Class

class LinearModel:

    def __init__(self, dim):
        """
        w : coef, array (d,)
        c : intercept, array (1,)
        """
        self.dim = dim

        self.params = {
            'coef': np.random.randn(dim), # shape (d,)
            'intercept': np.random.randn(1) # shape (1,)
        }

        self.cache = {}


    def forward(self, x, override_params = None):
        """
        Calculate y_pred given x and the current params, then update the cache.
        - d : dimension of parameters coef
        - N : number of observations in x, ie. batch_size
        - x : coef x nbr of observations
        - override_params : dict[str, np.ndarray] or None. If provided, use these parameters instead of self.params.
        """

        # Use override parameters if provided (for look-ahead, eg. Nesterov)
        if override_params is not None:
            params = override_params
        else:
            params = self.params
    
        if x.ndim == 1:
            x = x.reshape(-1,self.dim) # shape (N,d)

        y_pred = x @ params['coef']  + params['intercept'] # shape (N,)

        # Update the cache only if we are not override_params, ie not for look-ahead.
        if override_params is None:
            self.cache['x'] = x 
            self.cache['y_pred'] = y_pred 

        return y_pred 
    


    def backward(self, y_true, loss, override_params = None):
        """
        Return the gradient of the loss with respect to the params. Can handle override_params for look-ahead.
        """
            
        x = self.cache['x']

        # to handle look-ahead
        if override_params is not None:
            y_pred = x @ override_params['coef'] + override_params['intercept']
        else:
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