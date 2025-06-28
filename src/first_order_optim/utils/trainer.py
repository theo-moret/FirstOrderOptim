# Import 

import numpy as np
import logging  

from first_order_optim.model.base import BaseModel
from first_order_optim.loss.base import BaseLoss
from first_order_optim.optimizer.base import BaseOptimizer
from first_order_optim.scheduler.base import BaseScheduler

# Class

class Trainer():

    def __init__(self, model: BaseModel, loss: BaseLoss, optimizer: BaseOptimizer, n_epochs: int, batch_size: int, scheduler: BaseScheduler = None):

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.loss_cache = [[] for k in range(n_epochs)]
        
        self.grad_cache = {}
        for param in self.model.params:
            self.grad_cache[param] = [[] for k in range(n_epochs)]


    def train(self, X: np.ndarray, Y: np.ndarray):
        """
        Set up a training loop for the model given data (X,Y), going through the data set for n_epochs, each divided into n_batches after shuffling at the beggining of each epoch. 
        Schematically does : For epoch -> Shuffle -> For bacth -> Make prediction -> Calculate loss + gradients of the loss wrt params -> Take optimizer step -> Update model -> Take scheduler step. 

        Args:

        - X (np.ndarray of shape (N,d)): N being the number of observations and d the dimension of covariates.
        - Y (np.ndarray of shape (N,p)): N being the number of observations and p the dimension of the label. 
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

                # Update the learning rate via the scheduler if provided
                if self.scheduler is not None:
                    self.scheduler.step()

                # Update loss and gradient caches
                self.loss_cache[epoch].append(batch_loss)
                for param in self.model.params:
                    self.grad_cache[param][epoch].append(grads[param])

            avg_loss = epoch_loss / n_batches
            logging.info(f"Epoch {epoch+1}/{self.n_epochs} - Loss: {avg_loss:.4f}")

        return None
    


    def get_loss_cache(self):
        """
        Return the loss cache.
        """
        return np.array(self.loss_cache)
    
    def get_grad_cache(self):
        """
        Return the grad cache.
        """
        return self.grad_cache





# Main

if __name__ == "__main__":
    pass 

