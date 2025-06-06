
# Import 

from .base import BaseOptimizer


# Class

class GradientDescent(BaseOptimizer):

    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update(self, params, grads):
        lr = self.lr
        for key in params:
            params[key] = params[key] - lr * grads[key]


if __name__ == "__main__":
    pass