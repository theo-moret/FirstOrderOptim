# Import 

import numpy as np

from utils.trainer import Trainer
from losses.mse import MSELoss
from models.linear_model import LinearModel
from optimizers.adagrad import AdaGrad

# Main

if __name__ == '__main__':
    np.random.seed(1)

    # Generate data
    w, c = np.array([4., 2., -10.]), -2
    n = 10000
    X = np.random.random_sample((n, 3))
    noise = np.random.normal(0.0, 0.5, (n,))
    Y = X @ w + c + noise

    # Initialize model, loss, optimizer
    model = LinearModel(dim=3)
    loss = MSELoss()
    optimizer = AdaGrad(learning_rate=0.2)

    # Training loop
    epochs = 10
    batch_size = 100
    trainer = Trainer(model, loss, optimizer, epochs, batch_size)
    trainer.train(X, Y)

    print(f"True params : (w,c) = ({w},{c})")
    print(f"Estimated params : (w_esti, c_esti) = ({model.params['coef']},{model.params['intercept']})")