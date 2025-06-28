# Import 

import numpy as np

from first_order_optim.utils import Trainer
from first_order_optim.loss import MSELoss
from first_order_optim.model import LinearModel
from first_order_optim.optimizer import AdaGrad

# Main

if __name__ == '__main__':
    
    np.random.seed(1)

    # Generate sparse data
    d = 100
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
    optimizer = AdaGrad(learning_rate=0.5)

    # Training loop
    epochs = 100
    batch_size = 100
    trainer = Trainer(model, loss, optimizer, epochs, batch_size)
    trainer.train(X, Y)

    # True vs estimated weights
    w_esti = model.params['coef']    
    c_esti = model.params['intercept']

    # Compute MSE for weights and bias 
    mse_w = np.mean((w_esti - w)**2).item()
    print(f"MSE on w: {mse_w:.4f}")
    bias_error = abs(c_esti - c).item()
    print(f"Error on c (bias): {bias_error:.4f}")


