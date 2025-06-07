
# Import 

from .base import BaseOptimizer


# Class

class MomentumHeavyBall(BaseOptimizer):

    def __init__(self, learning_rate, gamma):
        super().__init__(learning_rate)
        self.gamma = gamma
        self.m_cache = {}

    def step(self, params, grads):

        lr = self.lr
        gamma = self.gamma


        # Setup first value to grad only
        if len(self.m_cache) == 0:
            for key in params:
                self.m_cache[key] = grads[key]

        # Update momentum value
        else:
            for key in params:
                self.m_cache[key] =  (gamma * self.m_cache[key] + (1-gamma)*grads[key])
        
        for key in params:
            params[key] = params[key] - lr * self.m_cache[key]
        
        return params


if __name__ == "__main__":
    pass