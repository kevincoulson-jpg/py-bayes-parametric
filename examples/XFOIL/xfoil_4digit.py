import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from XFOIL.xfoil_bayes import XFoilBayes

def max_cdcl_objective(opt_obj):
    clcd = opt_obj.cl / opt_obj.cd if opt_obj.cd > 0 else np.nan
    return clcd

def max_t_objective(opt_obj):
    return opt_obj.x[2]

if __name__ == "__main__":
    variables = [
        {'input_name': 'm', 'input_bounds': [0, 0.06]},
        {'input_name': 'p', 'input_bounds': [0.1, 0.6]},
        {'input_name': 't', 'input_bounds': [0.06, 0.2]},
    ]
    objectives = [max_cdcl_objective, max_t_objective]

    opt = XFoilBayes(variables, objectives, n_init=5, n_iter=35)
    predefined_x = [0.03, 0.3, 0.13]
    print(opt.evaluate_objectives(predefined_x))

    opt.optimize()
    opt.output_npz(x_labels=["m", "p", "t"], y_labels=["cl/cd", "t"], y_weights=[1, 1], filename="C:/Users/kevin/Documents/py-bayes-parametric/examples/XFOIL/data/xfoil_4digit.npz")