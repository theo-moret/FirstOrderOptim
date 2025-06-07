
# Import 

from .base import BaseOptimizer


# Class

class Momentum(BaseOptimizer):

    def __init__(self, learning_rate, gamma):
        super().__init__(learning_rate)
        self.gamma = gamma
        self.velocity = {}

    def step(self, params, grads):

        lr = self.lr
        gamma = self.gamma


        # Setup first value to grad only
        if len(self.velocity) == 0:
            for key in params:
                self.velocity[key] = grads[key]

        # Update momentum value
        else:
            for key in params:
                self.velocity[key] =  gamma * self.velocity[key] + grads[key]
        
        for key in params:
            params[key] = params[key] - lr * self.velocity[key]
        
        return params


if __name__ == "__main__":
    pass