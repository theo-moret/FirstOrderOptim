# Import 

from first_order_optim.scheduler.base import BaseScheduler
from first_order_optim.optimizer.base import BaseOptimizer

# Class 


class DecayRateScheduler(BaseScheduler):
    """
    Time–based (inverse) learning-rate decay eta_t = eta_0 / (1 + k*t) with:
        t = nbr of steps
        k = decay rate, positive number
    """

    def __init__(self, decay_rate: float):
        super().__init__()
        self.decay_rate = decay_rate
        self.eta0 = self.optimizer.lr
        self.time_step = 0


    def step(self):
        """Update the learning rate as eta_0/(1+kt), and update the learning rate of the optimizer in place."""
        self.time_step += 1
        self.optimizer.lr  = self.eta0 / (1+self.decay_rate * self.time_step)


# Main 

if __name__ == "__main__":
    pass