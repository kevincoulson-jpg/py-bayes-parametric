import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from general_bayes_opt import GeneralBayesOpt
from xfoil_interface import evaluate_custom_design
from freecad_interface import FreeCADInterface
from gen_airfoil import format_airfoil_coordinates, read_dxf_airfoil, organize_airfoil_coordinates

class CustomFoilBayes(GeneralBayesOpt):
    def __init__(self, model_path, variables, objective_functions, constraints, reynolds_number: int=3000000, alpha: float=5.0, n_init=5, n_iter=20, device='cpu'):
        """
        variables: list of dicts with keys 'input_name', 'input_bounds'
        objective_functions: list of callables, each takes (interface) and returns a float
        """
        self.model_path = model_path
        self.interface = FreeCADInterface()
        self.interface.set_document(self.model_path)
        self.variables = variables
        self.objective_functions = objective_functions
        self.constraints = constraints
        self.reynolds_number = reynolds_number
        self.alpha = alpha
        self.cl, self.cd = None, None

        # Fix bounds construction for BoTorch
        lows = [v['constraint_bounds'][0] for v in variables]
        highs = [v['constraint_bounds'][1] for v in variables]
        bounds = [lows, highs]
        print(f"[INIT] Bounds: {bounds}")  # Debug print
        super().__init__(bounds, self.evaluate_objectives, constraints, n_init=n_init, n_iter=n_iter, device=device)

    def update(self, x):
        # Set all parameters in FreeCAD, run FEA, and export results
        print(f"[UPDATE] x={x}")
        for var, value in zip(self.variables, x):
            self.interface.modify_parameter(var['object_name'], var['constraint_name'], value)
        self.interface.export_dxf()

        ## TODO: CLEAN UP FILE PATH MADNESS HERE
        x_upper, y_upper, x_lower, y_lower = read_dxf_airfoil(dxf_file="C:/Users/kevin/Documents/bayes-foil/data/foil.dxf")
        x_ordered, y_ordered = organize_airfoil_coordinates(x_upper, y_upper, x_lower, y_lower)
        format_airfoil_coordinates(x_ordered, y_ordered, output_file="C:/Users/kevin/Documents/bayes-foil/data/airfoil.dat") 
        dat_filepath = r"C:\Users\kevin\Documents\bayes-foil\data\airfoil.dat"

        cl, cd = evaluate_custom_design(dat_filepath, self.reynolds_number, self.alpha)
        self.cl, self.cd = cl, cd

    def check_constraints(self, x):
        """Check if all constraints are satisfied.
        Changed from original implementation to pass self to constraint functions.
        """
        print(f"[CHECK CONSTRAINTS] x={x}")
        return all(constraint(self) for constraint in self.constraints)
    
    def evaluate_objectives(self, x):
        print(f"[EVAL XFoil] x={x}")
        # Update FreeCAD state using full_workflow
        self.x = x  # in case the usual optimization workflow isn't used
        self.update(x)

        # Evaluate all objectives using self
        results = [obj_func(self) for obj_func in self.objective_functions]
        print(f"[EVAL XFoil] objectives={results}")
        return results