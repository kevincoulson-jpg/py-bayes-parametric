import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from XFOIL.custom_foil_bayes import CustomFoilBayes

def max_cdcl_objective(opt_obj):
    clcd = opt_obj.cl / opt_obj.cd if opt_obj.cd > 0 else np.nan
    return clcd

if __name__ == "__main__":
    variables = [
        {'object_name': 'Sketch', 'constraint_name': 'x1U', 'constraint_bounds': [300, 800]},
        {'object_name': 'Sketch', 'constraint_name': 'x1L', 'constraint_bounds': [300, 800]},
        {'object_name': 'Sketch', 'constraint_name': 'x2U', 'constraint_bounds': [700, 1500]},
        {'object_name': 'Sketch', 'constraint_name': 'x2L', 'constraint_bounds': [700, 1500]},
        {'object_name': 'Sketch', 'constraint_name': 'y1U', 'constraint_bounds': [100, 500]},
        {'object_name': 'Sketch', 'constraint_name': 'y1L', 'constraint_bounds': [0, 400]},
        {'object_name': 'Sketch', 'constraint_name': 'y2U', 'constraint_bounds': [100, 500]},
        {'object_name': 'Sketch', 'constraint_name': 'y2L', 'constraint_bounds': [0, 400]},
        {'object_name': 'Sketch', 'constraint_name': 'yTE', 'constraint_bounds': [400, 600]},
    ]
    objectives = [max_cdcl_objective]
    model_path = "C:/Users/kevin/Documents/bayes-foil/data/foil.FCStd"
    opt = CustomFoilBayes(model_path, variables, objectives, n_init=5, n_iter=195)

    predefined_x = [300, 300, 1200, 700, 200, 300, 200, 300, 400]
    assert len(predefined_x) == len(variables)
    print("Testing update with predefined x:", predefined_x)
    results = opt.evaluate_objectives(predefined_x)
    print(results)

    opt.optimize()
    opt.output_npz(x_labels=["x1U", "x1L", "x2U", "x2L", "y1U", "y1L", "y2U", "y2L", "yTE"], y_labels=["cl/cd"], y_weights=[1], filename="C:/Users/kevin/Documents/py-bayes-parametric/examples/XFOIL/data/control_point_foil.npz")