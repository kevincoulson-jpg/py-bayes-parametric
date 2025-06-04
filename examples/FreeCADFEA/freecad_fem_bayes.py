import numpy as np
from freecad_interface import FreeCADInterface
from general_bayes_opt import GeneralBayesOpt

class FreeCADFEMBayes(GeneralBayesOpt):
    def __init__(self, model_path, variables, objective_functions, n_init=5, n_iter=20, device='cpu'):
        """
        model_path: path to FreeCAD model
        variables: list of dicts with keys 'object_name', 'constraint_name', 'constraint_bounds'
        objective_functions: list of callables, each takes (interface) and returns a float
        """
        self.model_path = model_path
        self.variables = variables
        self.interface = FreeCADInterface()
        self.interface.set_document(model_path)
        self.objective_functions = objective_functions
        # Fix bounds construction for BoTorch
        lows = [v['constraint_bounds'][0] for v in variables]
        highs = [v['constraint_bounds'][1] for v in variables]
        bounds = [lows, highs]
        print(f"[INIT] Bounds: {bounds}")  # Debug print
        super().__init__(bounds, self.evaluate_objectives, n_init=n_init, n_iter=n_iter, device=device)

    def update(self, x):
        # Set all parameters in FreeCAD, run FEA, and export results
        print(f"[UPDATE] x={x}")
        self.interface.set_document(self.model_path)
        
        # Update x_values dictionary
        self.interface.x_values = {
            f"{var['object_name']}::{var['constraint_name']}": value 
            for var, value in zip(self.variables, x)
        }
        
        # Update FreeCAD parameters
        for var, value in zip(self.variables, x):
            self.interface.modify_parameter(var['object_name'], var['constraint_name'], value)
        # Run the full workflow line-by-line
        self.interface.recompute_mesh_fem()
        self.interface.run_fem()
        print("FEM run complete")
        self.interface.export_results_vtu()
        print("VTU export complete")
        self.interface.initiate_vtk_results()
        print("VTK results initiated")

    def evaluate_objectives(self, x):
        print(f"[EVAL FREECAD] x={x}")
        # Update FreeCAD state using full_workflow
        print(x)
        self.update(x)

        # Evaluate all objectives using self.interface
        results = [obj_func(self.interface) for obj_func in self.objective_functions]
        print(f"[EVAL FREECAD] objectives={results}")
        return results