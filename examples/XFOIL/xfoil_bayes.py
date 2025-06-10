import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from general_bayes_opt import GeneralBayesOpt
from xfoil_interface import evaluate_naca_design

class XFoilBayes(GeneralBayesOpt):
    def __init__(self, variables, objective_functions, reynolds_number: int=3000000, alpha: float=5.0, n_init=5, n_iter=20, device='cpu'):
        """
        variables: list of dicts with keys 'input_name', 'input_bounds'
        objective_functions: list of callables, each takes (interface) and returns a float
        """
        self.variables = variables
        self.objective_functions = objective_functions
        self.reynolds_number = reynolds_number
        self.alpha = alpha
        self.x, self.cl, self.cd = None, None, None

        # Fix bounds construction for BoTorch
        lows = [v['input_bounds'][0] for v in variables]
        highs = [v['input_bounds'][1] for v in variables]
        bounds = [lows, highs]
        print(f"[INIT] Bounds: {bounds}")  # Debug print
        super().__init__(bounds, self.evaluate_objectives, n_init=n_init, n_iter=n_iter, device=device)

    def update(self, x):
        # Set all parameters in FreeCAD, run FEA, and export results
        print(f"[UPDATE] x={x}")
        cl, cd = evaluate_naca_design(x[0], x[1], x[2], self.alpha, self.reynolds_number)
        self.x, self.cl, self.cd = x, cl, cd

    def evaluate_objectives(self, x):
        print(f"[EVAL XFoil] x={x}")
        # Update FreeCAD state using full_workflow
        print(x)
        self.update(x)

        # Evaluate all objectives using self
        results = [obj_func(self) for obj_func in self.objective_functions]
        print(f"[EVAL XFoil] objectives={results}")
        return results