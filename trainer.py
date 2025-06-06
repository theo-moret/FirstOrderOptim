# Import 

import numpy as np
from losses.mse import MSELoss
from models.linear_model import LinearModel
from optimizers.gradient_descent import GradientDescent

# Main

if __name__ == "__main__":

    np.random.seed(1)

    # Generate data

    w, c = 4, -2 # True params
    n = 10000 # Number of points 
    X = np.random.random_sample((n,))
    noise = np.random.normal(0.0, 0.2, (n,))
    Y = w * X + c + noise


    # Initialize model, loss, and optimizer

    model = LinearModel(dim=1)
    loss = MSELoss()
    optimizer = GradientDescent(learning_rate=0.2)

    # Training loop

    epochs = 200
    batch_size = 100
    num_batches = n // batch_size

    for epoch in range(epochs):
        for i in range(num_batches):
            # Fetch data
            x = X[i * batch_size: (i+1) * batch_size]
            y = Y[i * batch_size: (i+1) * batch_size]

            # Make prediction
            y_pred = model.forward(x)

            # Check loss
            if i == 0:
                print(f"Epoch {epoch} - Loss : {loss.forward(y_pred, y)}")

            # Obtain gradient of the loss wrt to params
            grads = model.backward(y, loss)

            # Take gradient step
            new_params = optimizer.step(model.params, grads)

            # Update model params
            model.update(new_params)

    # Show result 
    print(f"True params : (w,c) = ({w},{c})")
    print(f"\n Estimated params : (w_esti, c_esti) = ({model.params['coef']},{model.params['intercept']})")