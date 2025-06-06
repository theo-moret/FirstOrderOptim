
# Import 

from .base import BaseOptimizer


# Class

class GradientDescent(BaseOptimizer):

    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def step(self, params, grads):
        lr = self.lr
        new_params = {}
        for key in params:
            new_params[key] = params[key] - lr * grads[key]
        
        return new_params


if __name__ == "__main__":
    pass