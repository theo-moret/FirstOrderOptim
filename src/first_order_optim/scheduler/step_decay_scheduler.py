# Import 

from first_order_optim.scheduler.base import BaseScheduler
from first_order_optim.optimizer.base import BaseOptimizer

# Class 


class StepDecayScheduler(BaseScheduler):
    """
    Step decay learning-rate every n steps : eta_t = eta_{t-n} / gamma ie every n steps, drop by gamma the learning rate.
        n = step size
        gamma = drop factor
    """

    def __init__(self, optimizer: BaseOptimizer, step_size: int, gamma: float):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.time_step = 0


    def step(self):
        """Update the learning rate every n steps as eta_t = eta_{t-n} / gamma, and update the learning rate of the optimizer in place."""
        self.time_step += 1
        if self.time_step % self.step_size == 0:
            current_lr = self.optimizer.lr
            new_lr = current_lr/self.gamma
            self.optimizer.lr = new_lr
        

# Main 

if __name__ == "__main__":
    pass