# Import 

import numpy as np


# Class

class LinearModel:

    def __init__(self, w, c):
        """
        w : coef, array (d,)
        c : intercept, float
        """

        self.params = {
            'coef': w, # shape (d,)
            'intercept': c
        }
        self.cache = {}


    def forward(self, x):
        """
        Calculate y_pred given x and the current params, then update the cache.
        """

        y_pred = x @ self.params['coef']  + self.params['intercept'] 

        self.cache['x'] = x # shape (N,d)
        self.cache['y_pred'] = y_pred # shape (N,)

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




if __name__ == "__main__":
    pass