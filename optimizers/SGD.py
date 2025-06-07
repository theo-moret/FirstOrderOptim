
# Import 

from .base import BaseOptimizer


# Class

class SGD(BaseOptimizer):

    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def step(self, params, grads):
        lr = self.lr
        
        for key in params:
            params[key] = params[key] - lr * grads[key]
        
        return params


if __name__ == "__main__":
    pass