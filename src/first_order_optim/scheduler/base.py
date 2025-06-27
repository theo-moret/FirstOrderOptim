# Import 

from abc import ABC, abstractmethod

from first_order_optim.optimizer.base import BaseOptimizer


# Class

class BaseScheduler(ABC):
    """
    Base for all scheduler objects.

    Subclasses must implement 'step'.
    """


    def __init__(self, optimizer: BaseOptimizer):
        self.optimizer = optimizer 

    @abstractmethod
    def step(self):
        """Implement the step method to adjust the current learning rate of the optimizer in place."""
        
    