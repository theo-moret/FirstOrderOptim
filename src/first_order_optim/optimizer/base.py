# Class

class BaseOptimizer:

    def __init__(self, learning_rate):
        self.lr = learning_rate

    def step(self, params, grads):
        raise NotImplementedError("Update method not implemented.")


if __name__ == "__main__":
    pass