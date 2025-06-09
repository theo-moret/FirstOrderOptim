# Import 

import numpy as np


# Class

class BoothFunction:
    
    def __init__(self, x0, y0):

        self.params = {
            'x' : x0,
            'y' : y0
        }

        return None

    def forward(self):
        """
        Return the value of the Booth Function evaluated at (x,y)
        """

        x = self.params['x']
        y = self.params['y']

        f_value = (x + 2*y - 7) ** 2 + (2*x + y - 5) ** 2

        return f_value 
    


    def backward(self):
        """
        Return the gradient of the Booth function wrt to each parameter x and y.
        """
            
        x = self.params['x']
        y = self.params['y']

        dfdx = 10*x + 8*y - 34
        dfdy = 8*x + 10*y -38

        return {'x': dfdx, 'y': dfdy}


    def update(self, new_params):
        """
        Update model params to new_params.
        """
        self.params = new_params
        return None
    

class ThreeHumpCamel:
    
    def __init__(self, x0, y0):

        self.params = {
            'x' : x0,
            'y' : y0
        }

        return None

    def forward(self):
        """
        Return the value of the Booth Function evaluated at (x,y)
        """

        x = self.params['x']
        y = self.params['y']

        f_value = 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2

        return f_value 
    


    def backward(self):
        """
        Return the gradient of the Booth function wrt to each parameter x and y.
        """
            
        x = self.params['x']
        y = self.params['y']

        dfdx = 4*x - 4.2*x**3 + x**5 + y
        dfdy = x + 2*y

        return {'x': dfdx, 'y': dfdy}


    def update(self, new_params):
        """
        Update model params to new_params.
        """
        self.params = new_params
        return None 
    
if __name__ == "__main__":
    pass