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
    d = 1000
    n = 100000
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

    # true vs estimated weights
    w_est = model.params['coef']    
    c_est = model.params['intercept']

    # compute percent‐error 
    weight_pct_errors = np.abs(w_est - w) / np.abs(w) * 100
    intercept_pct_error = np.abs(c_est - c) / np.abs(c) * 100
    all_pct_errors = np.concatenate([weight_pct_errors, intercept_pct_error])

    # average
    avg_pct_error = np.mean(all_pct_errors)

    print(f"Average percent error across parameters: {avg_pct_error:.2f}%")

