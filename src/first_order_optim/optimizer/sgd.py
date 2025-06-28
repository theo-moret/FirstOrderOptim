
# Import 

import numpy as np 

from first_order_optim.optimizer.base import BaseOptimizer

# Class

class SGD(BaseOptimizer):

    def __init__(self, learning_rate: float):
        super().__init__(learning_rate)

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ See BaseOptimizer.step, but implement the optimizer step and return dict of updated params. """
        lr = self.lr

        new_params = {}
        for key in params:
            new_params[key] = params[key] - lr * grads[key]
        
        return new_params


if __name__ == "__main__":
    pass