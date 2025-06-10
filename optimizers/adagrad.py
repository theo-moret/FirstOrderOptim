
# Import 

from .base import BaseOptimizer
import numpy as np

# Class

class AdaGrad(BaseOptimizer):

    def __init__(self, learning_rate=1e-2, eps=1e-8):
        super().__init__(learning_rate)
        self.eps = eps
        self.gradient_cache = {}

    def step(self, params, grads):
        """ Implement Adagrad optimizer step. """

        lr = self.lr
        eps = self.eps


        # Setup first value 
        if len(self.gradient_cache) == 0:
            for key in params:
                self.gradient_cache[key] = grads[key]**2

        # Update the gradient cache 
        else:
            for key in params:
                self.gradient_cache[key] += grads[key]**2

            
        new_params = {}
        for key in params:
            new_params[key] = params[key] - lr / np.sqrt(self.gradient_cache[key] + eps) * grads[key]
        
        return new_params


if __name__ == "__main__":
    pass