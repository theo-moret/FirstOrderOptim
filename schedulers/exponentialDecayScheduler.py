# Import 

import numpy as np

from baseScheduler import BaseScheduler
from ..optimizers.base import BaseOptimizer

# Class 


class ExpDecayScheduler(BaseScheduler):
    """
    Exponential learning-rate decay eta_t = eta_0 exp(-kt) with:
        t = nbr of steps
        k = decay rate 
    """

    def __init__(self, optimizer: BaseOptimizer, decay_rate: float):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.eta0 = optimizer.lr
        self.time_step = 0


    def step(self):
        """Update the learning rate as eta_0/(1+kt), and update the learning rate of the optimizer in place."""
        self.time_step += 1
        self.optimizer.lr  = self.eta0 * np.exp(-self.decay_rate * self.time_step)


# Main 

if __name__ == "__main__":
    pass