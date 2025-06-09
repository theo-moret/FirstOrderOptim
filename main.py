# Import 

import numpy as np
import matplotlib.pyplot as plt 


from utils.trainer import Trainer
from losses.mse import MSELoss
from models.linear_model import LinearModel
from optimizers.SGD import SGD
from optimizers.momentum import Momentum
from optimizers.NesterovMomentum import NesterovMomentum
from optimizers.adagrad import AdaGrad
from models.test_functions import BoothFunction

# Functions 

def booth_function(x, y):
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2



# Main

if __name__ == "__main__":

    np.random.seed(1)

    linear = False 
    booth = True 

    if linear == True :
        # Generate data

        w, c = np.array([4.,2., -10.]), -2 # True params
        n = 10000 # Number of points 
        X = np.random.random_sample((n,3))
        noise = np.random.normal(0.0, 0.5, (n,))
        Y = X @ w + c + noise


        # Initialize model, loss, and optimizer

        model = LinearModel(dim=3)
        loss = MSELoss()
        optimizer = AdaGrad(learning_rate=0.2)
        
        # Training loop
        
        epochs = 10
        batch_size = 100
        trainer = Trainer(model, loss, optimizer, epochs, batch_size)

        trainer.train(X,Y)

        # Show result 
        print(f"True params : (w,c) = ({w},{c})")
        print(f"Estimated params : (w_esti, c_esti) = ({model.params['coef']},{model.params['intercept']})")


    if booth == True:

        
        x0, y0 = -9, -9
        function = BoothFunction(x0, y0)

        optimizer = AdaGrad(learning_rate=1)

        n_steps = 20

        xs, ys = np.zeros(shape=(n_steps+1,)), np.zeros(shape=(n_steps+1,))
        xs[0], ys[0] = x0, y0

        for i in range(n_steps):

            print(f'Step {i+1} : {function.forward()}')

            grads = function.backward()

            new_params = optimizer.step(function.params, grads)

            function.update(new_params)

            xs[i+1], ys[i+1] = new_params['x'],new_params['y']

        print(f'Minimum estimated at ({function.params})')

        # Grid 
        x_vals = np.linspace(-10, 10, 400)
        y_vals = np.linspace(-10, 10, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = booth_function(X, Y)

        # Plot contours
        plt.figure(figsize=(8, 6))
        contours = plt.contour(X, Y, Z, levels=50, cmap='viridis')
        plt.clabel(contours, inline=True, fontsize=8)

        # Plot trajectory
        plt.plot(xs, ys, marker='o', color='red', label='Trajectoire')
        
        plt.title("Optimizer Trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        plt.show()