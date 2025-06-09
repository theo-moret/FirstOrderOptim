
# Import 

from .base import BaseOptimizer
import numpy as np

# Class

class NesterovMomentum(BaseOptimizer):

    def __init__(self, learning_rate, gamma):
        super().__init__(learning_rate)
        self.gamma = gamma
        self.velocity = {}

    def step(self, model, loss, Y):
        """
        Since Nesterov implies a look-ahead step and then compute the gradient, it needs to acces the model, the loss and the data (X,Y).
        """

        lr = self.lr
        gamma = self.gamma


        # Setup first value to 0
        if len(self.velocity) == 0:
            for key in model.params:
                self.velocity[key] = np.zeros_like(model.params[key])


        # Create look-ahead params that will be used to compute the gradient
        pseudo_params = {key: model.params[key] - gamma * self.velocity[key] for key in model.params}


        # Compute gradients with look-ahead params
        grads = model.backward(Y, loss, override_params = pseudo_params)

        # Update momentum value
        for key in model.params:
            self.velocity[key] = gamma * self.velocity[key] + lr * grads[key]
            model.params[key] -= self.velocity[key]
        
        return model.params


if __name__ == "__main__":
    pass