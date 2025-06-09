# Import 

import numpy as np

# Class

class Trainer():

    def __init__(self, model, loss, optimizer, n_epochs, batch_size):

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.loss_cache = {}


    def train(self, X, Y, print_loss = True):
        """
        Set up a training loop for the model given data (X,Y). The parameter print_loss allows for loss printing at the beginning of every epoch. 
        If the number of epoch is 1, then the losses are stocked into the cache for every batch.
        """

        model = self.model
        loss = self.loss
        optimizer = self.optimizer
        n = len(X)
        batch_size = self.batch_size
        n_batches = n // self.batch_size


        for epoch in range(self.n_epochs):
            for i in range(n_batches):
                # Fetch data
                x = X[i * batch_size: (i+1) * batch_size]
                y = Y[i * batch_size: (i+1) * batch_size]

                # Make prediction
                y_pred = model.forward(x)

                # Loss value evaluation
                loss_value = loss.forward(y_pred, y)
                
                # Print and stock loss value for evey batch if the number of epoch is 1
                if self.n_epochs == 1:
                    print(f"Batch {i+1} - Loss : {loss_value}")
                    self.loss_cache[i +1] = loss_value

                # Check loss at beginning of epoch and stock it in the cache
                elif i == 0 and print_loss == True:
                    print(f"Epoch {epoch + 1} - Loss : {loss_value}")
                    self.loss_cache[epoch + 1] = loss_value

                # Obtain gradient of the loss wrt to params
                grads = model.backward(y, loss)

                # Take gradient step, check if Nesterov method to handle look-ahead
                if optimizer.__class__.__name__ == 'NesterovMomentum':
                    new_params = optimizer.step(model, loss, y)
                
                else:
                    new_params = optimizer.step(model.params, grads)

                # Update model params
                model.update(new_params)


        return None
    


    def get_losses(self):
        """
        Return the loss values for every epoch (or batch if the number of epoch is 1) in a numpy array.
        """

        loss_values = np.fromiter(self.loss_cache.values(), dtype=float)
        return loss_values




# Main

if __name__ == "__main__":
    pass 

