# Import 

import numpy as np
import logging  


# Log Setting 

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

# Class

class Trainer():

    def __init__(self, model, loss, optimizer, n_epochs, batch_size):

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.loss_cache = {}


    def train(self, X, Y):
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
            
            # Shuffle data at the start of each epoch
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            
            epoch_loss = 0

            for i in range(n_batches):
                # Fetch data
                start = i * batch_size
                end = min((i+1) * batch_size, n)
                x_batch = X_shuffled[start:end]
                y_batch = Y_shuffled[start:end]

                # Make prediction
                y_pred = model.forward(x_batch)

                # Loss value evaluation
                batch_loss = loss.forward(y_pred, y_batch)
                epoch_loss += batch_loss

                # Obtain gradient of the loss wrt to params
                grads = model.backward(y_batch, loss)

                # Take gradient step, check if Nesterov method to handle look-ahead
                if optimizer.__class__.__name__ == 'NesterovMomentum':
                    if len(optimizer.velocity)==0:
                        for key in model.params:
                            optimizer.velocity[key] = np.zeros_like(model.params[key])
                    lookahead_params = {k: model.params[k] - optimizer.gamma * optimizer.velocity[k] for k in model.params}
                    grads = model.backward(y_batch, loss, override_params=lookahead_params)
                    new_params = optimizer.step(model.params, grads)
                
                else:
                    new_params = optimizer.step(model.params, grads)

                # Update model params
                model.update(new_params)

            avg_loss = epoch_loss / n_batches
            logging.info(f"Epoch {epoch+1}/{self.n_epochs} - Loss: {avg_loss:.4f}")

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

