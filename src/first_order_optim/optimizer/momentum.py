
# Import 

import numpy as np

from first_order_optim.optimizer.base import BaseOptimizer

# Class

class Momentum(BaseOptimizer):

    def __init__(self, learning_rate: float, gamma: float):
        super().__init__(learning_rate)
        self.gamma = gamma
        self.velocity = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ See BaseOptimizer.step, but implement the optimizer step and return dict of updated params. """
        lr = self.lr
        gamma = self.gamma


        # Setup first value to grad only
        if len(self.velocity) == 0:
            for key in params:
                self.velocity[key] = grads[key]

        # Update momentum value
        else:
            for key in params:
                self.velocity[key] =  gamma * self.velocity[key] + (1-gamma)*grads[key]
        
        new_params = {}
        for key in params:
            new_params[key] = params[key] - lr * self.velocity[key]
        
        return new_params


if __name__ == "__main__":
    pass