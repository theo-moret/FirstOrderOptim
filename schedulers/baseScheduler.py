# Class

class BaseScheduler:

    def __init__(self, optimizer):
        self.optimizer = optimizer 

    def step(self):
        """Implement the step method to adjust the current learning rate."""
        pass
    