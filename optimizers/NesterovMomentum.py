
# Import 

from .base import BaseOptimizer
import numpy as np

# Class

class NesterovMomentum(BaseOptimizer):

    def __init__(self, learning_rate, gamma):
        super().__init__(learning_rate)
        self.gamma = gamma
        self.velocity = {}

    def step(self, params, grads):
        """        """

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