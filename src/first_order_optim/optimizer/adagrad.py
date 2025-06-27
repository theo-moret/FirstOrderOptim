
# Import 

import numpy as np

from first_order_optim.optimizer.base import BaseOptimizer

# Class

class AdaGrad(BaseOptimizer):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gradient_cache = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ See BaseOptimizer.step, but implement the optimizer step and return dict of updated params. """

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