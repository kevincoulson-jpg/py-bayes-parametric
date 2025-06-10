import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from XFOIL.custom_foil_bayes import CustomFoilBayes

def max_cdcl_objective(opt_obj):
    clcd = opt_obj.cl / opt_obj.cd if opt_obj.cd > 0 else np.nan
    if clcd > 200:
        print(f"clcd > 200: {clcd}, non-converged results")
        return -1e15
    elif clcd < -20:
        print(f"clcd < -20: {clcd}, non-converged results")
        return -1e15
    return clcd

def constraint_x1U_lt_x2U(opt_obj): 
    print(f"[CHECK CONSTRAINTS] x1U_lt_x2U")
    x1U = opt_obj.x[0]
    x2U = opt_obj.x[2]
    if x1U > x2U:
        print(f"x1U > x2U: {x1U}, {x2U}")
    return x1U < x2U

def constraint_x1L_lt_x2L(opt_obj):
    print(f"[CHECK CONSTRAINTS] x1L_lt_x2L")
    x1L = opt_obj.x[1]
    x2L = opt_obj.x[3]
    if x1L > x2L:
        print(f"x1L > x2L: {x1L}, {x2L}")
    return x1L < x2L

def constraint_y1L_lt_y2L(opt_obj):
    print(f"[CHECK CONSTRAINTS] y1L_lt_y2L")
    y1L = opt_obj.x[5]
    y2L = opt_obj.x[7]
    if y1L > y2L:
        print(f"y1L > y2L: {y1L}, {y2L}")
    return y1L < y2L

def constraint_y1U_gt_y2U(opt_obj):
    print(f"[CHECK CONSTRAINTS] y1U_gt_y2U")
    y1U = opt_obj.x[4]
    y2U = opt_obj.x[6]
    if y1U < y2U:
        print(f"y1U < y2U: {y1U}, {y2U}")
    return y1U > y2U

if __name__ == "__main__":
    variables = [
        {'object_name': 'Sketch', 'constraint_name': 'x1U', 'constraint_bounds': [300, 800]},
        {'object_name': 'Sketch', 'constraint_name': 'x1L', 'constraint_bounds': [300, 800]},
        {'object_name': 'Sketch', 'constraint_name': 'x2U', 'constraint_bounds': [700, 1500]},
        {'object_name': 'Sketch', 'constraint_name': 'x2L', 'constraint_bounds': [700, 1500]},
        {'object_name': 'Sketch', 'constraint_name': 'y1U', 'constraint_bounds': [100, 500]},
        {'object_name': 'Sketch', 'constraint_name': 'y1L', 'constraint_bounds': [0, 500]},
        {'object_name': 'Sketch', 'constraint_name': 'y2U', 'constraint_bounds': [100, 500]},
        {'object_name': 'Sketch', 'constraint_name': 'y2L', 'constraint_bounds': [0, 500]},
        {'object_name': 'Sketch', 'constraint_name': 'yTE', 'constraint_bounds': [400, 600]},
    ]
    objectives = [max_cdcl_objective]
    constraints = [constraint_x1U_lt_x2U, constraint_x1L_lt_x2L, constraint_y1U_gt_y2U, constraint_y1L_lt_y2L]
    model_path = "C:/Users/kevin/Documents/bayes-foil/data/foil.FCStd"
    opt = CustomFoilBayes(model_path, 
                          variables=    variables, 
                          objective_functions=objectives, 
                          constraints=constraints, 
                          reynolds_number=3000000, 
                          alpha=3, 
                          n_init=15, 
                          n_iter=485)

    predefined_x = [600, 400, 1200, 700, 200, 300, 200, 300, 400]
    assert len(predefined_x) == len(variables)
    print("Testing update with predefined x:", predefined_x)
    results = opt.evaluate_objectives(predefined_x)
    # print(results)

    opt.optimize()
    opt.output_npz(x_labels=["x1U", "x1L", "x2U", "x2L", "y1U", "y1L", "y2U", "y2L", "yTE"], y_labels=["cl/cd"], y_weights=[1], filename="C:/Users/kevin/Documents/py-bayes-parametric/examples/XFOIL/data/control_point_foil.npz")