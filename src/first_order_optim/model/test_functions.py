# Import 

import numpy as np

from first_order_optim.model.base import BaseModel

# Class

class BoothFunction(BaseModel):
    
    def __init__(self, x0: float, y0: float):
        """
        Set the starting points at (x0,y0).
        """
        super().__init__()
        self.params['x'] = x0
        self.params['y'] = y0
     

    def forward(self) -> float:
        """
        Return the value of the Booth Function evaluated at its current (x,y).
        """

        x = self.params['x']
        y = self.params['y']

        f_value = (x + 2*y - 7) ** 2 + (2*x + y - 5) ** 2

        return f_value 
    


    def backward(self) -> dict[str, float]:
        """
        Return the gradient of the Booth function wrt to each parameter x and y.
        """
            
        x = self.params['x']
        y = self.params['y']

        dfdx = 10*x + 8*y - 34
        dfdy = 8*x + 10*y -38

        return {'x': dfdx, 'y': dfdy}


    def evaluate (self, x: float, y: float) -> float:
        """Return the value of the function at its current (x,y)."""
        return (x + 2*y - 7) ** 2 + (2*x + y - 5) ** 2
    



class ThreeHumpCamel(BaseModel):
    
    def __init__(self, x0: float, y0: float):
        """
        Set the starting points at (x0,y0).
        """
        super().__init__()
        self.params['x'] = x0
        self.params['y'] = y0
     


    def forward(self) -> float:
        """
        Return the value of the Booth Function evaluated at its current (x,y).
        """

        x = self.params['x']
        y = self.params['y']

        f_value = 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2

        return f_value 
    


    def backward(self) -> dict[str, float]:
        """
        Return the gradient of the Booth function wrt to each parameter x and y.
        """
            
        x = self.params['x']
        y = self.params['y']

        dfdx = 4*x - 4.2*x**3 + x**5 + y
        dfdy = x + 2*y

        return {'x': dfdx, 'y': dfdy}


    def evaluate(self, x: float, y: float) -> float:
        """Return the value of the function at its current (x,y)."""
        return 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2


    
if __name__ == "__main__":
    pass