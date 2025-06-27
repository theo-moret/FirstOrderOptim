
# Import 

import numpy as np

from first_order_optim.optimizer.base import BaseOptimizer

# Class

class NesterovMomentum(BaseOptimizer):

    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.velocity = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ See BaseOptimizer.step, but implement the optimizer step and return dict of updated params. """
        
        lr = self.lr
        gamma = self.gamma


        # Setup first value to 0
        if len(self.velocity) == 0:
            for key in params:
                self.velocity[key] = np.zeros_like(params[key])


        new_params = {}
        for key in params:
            v_prev = self.velocity[key]
            self.velocity[key] = gamma * self.velocity[key] + lr * grads[key]
            new_params[key] = params[key] - gamma * v_prev + lr * grads[key]

        return new_params
        

if __name__ == "__main__":
    pass