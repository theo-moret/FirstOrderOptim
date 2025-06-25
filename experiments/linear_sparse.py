# Import 

import numpy as np

from utils.trainer import Trainer
from losses.mse import MSELoss
from models.linear_model import LinearModel
from optimizers.NesterovMomentum import NesterovMomentum
from optimizers.adagrad import AdaGrad
from schedulers.decayRateScheduler import DecayRate

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
    sch = DecayRate(optimizer=optimizer, decay_rate=0.01)

    # Training loop
    epochs = 100
    batch_size = 100
    trainer = Trainer(model, loss, optimizer, epochs, batch_size, scheduler=sch)
    trainer.train(X, Y)

    print("Norm between true and estimated weights:", np.linalg.norm(w - model.params['coef']))