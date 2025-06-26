# Import 

import numpy as np

from first_order_optim.utils import Trainer
from first_order_optim.loss import MSELoss
from first_order_optim.model import LinearModel
from first_order_optim.optimizer import AdaGrad
from first_order_optim.scheduler import DecayRateScheduler

# Main

if __name__ == '__main__':
    
    np.random.seed(1)

    # Generate sparse data
    d = 1000
    n = 10000
    indices = np.random.randint(0, d, size=n)
    X = np.eye(d)[indices]
    w = np.random.randn(d)
    c = -2.0
    noise = np.random.normal(0.0, 0.1, size=n)
    Y = w[indices] + c + noise

    # Initialize model, loss, optimizer
    model = LinearModel(dim=d)
    loss = MSELoss()
    optimizer = AdaGrad(learning_rate=0.2)
    sch = DecayRateScheduler(optimizer=optimizer, decay_rate=0.01)

    # Training loop
    epochs = 100
    batch_size = 100
    trainer = Trainer(model, loss, optimizer, epochs, batch_size, scheduler=sch)
    trainer.train(X, Y)

    print("Norm between true and estimated weights:", np.linalg.norm(w - model.params['coef']))