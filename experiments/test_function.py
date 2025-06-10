# Import 

import numpy as np
import matplotlib.pyplot as plt


from models.test_functions import ThreeHumpCamel
from optimizers.momentum import Momentum

# Main

if __name__ == '__main__':
    
    np.random.seed(1)

    ## Initialize function and optimizer
    x0, y0 = 1.8, -1.8
    function = ThreeHumpCamel(x0, y0)
    optimizer = Momentum(learning_rate=0.5, gamma=0.9)

    n_steps = 50
    xs, ys = np.zeros(n_steps + 1), np.zeros(n_steps + 1)
    xs[0], ys[0] = x0, y0

    for i in range(n_steps):
        print(f'Step {i+1} : {function.forward()}')
        grads = function.backward()
        new_params = optimizer.step(function.params, grads)
        function.update(new_params)
        xs[i+1], ys[i+1] = new_params['x'], new_params['y']

    print(f'Minimum estimated at ({function.params})')

    # Plotting
    x_vals = np.linspace(-2.01, 2.01, 100)
    y_vals = np.linspace(-2.01, 2.01, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = function.evaluate(X, Y)

    plt.figure(figsize=(8, 6))
    contours = plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.plot(xs, ys, marker='o', color='red', label='Trajectory')
    plt.title("Optimizer Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()