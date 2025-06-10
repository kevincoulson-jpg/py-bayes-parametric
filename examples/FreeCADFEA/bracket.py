import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from FreeCADFEA.freecad_fem_bayes import FreeCADFEMBayes

def max_stress_objective(interface):
    mesh = interface.vtk_mesh
    if mesh is not None and 'von Mises Stress' in mesh.point_data:
        return -np.max(mesh.point_data['von Mises Stress'])
    return np.nan

def min_volume_objective(interface):
    volume = float(interface.get_part_volume('Body'))
    return -volume

def max_displacement_objective(interface):
    mesh = interface.vtk_mesh
    if mesh is not None and 'Displacement Magnitude' in mesh.point_data:
        return -np.max(mesh.point_data['Displacement Magnitude'])
    return np.nan

if __name__ == "__main__":
    def degree_to_radian(degree):
        return degree * np.pi / 180
    
    variables = [
         {"object_name": "MainProfile", "constraint_name": "dBraceConnect", "constraint_bounds": [20, 30]},
         {"object_name": "MainProfile", "constraint_name": "tBracketH", "constraint_bounds": [4, 12]},
         {"object_name": "MainProfile", "constraint_name": "wBrace", "constraint_bounds": [4, 10]},
         {"object_name": "MainProfile", "constraint_name": "aBraceUpper", "constraint_bounds": [degree_to_radian(60), degree_to_radian(90)]},
         {"object_name": "MainProfile", "constraint_name": "aBraceLower", "constraint_bounds": [degree_to_radian(60), degree_to_radian(120)]},
         {"object_name": "MainProfile", "constraint_name": "wBraceD", "constraint_bounds": [5, 10]},
         {"object_name": "Pad001", "constraint_name": "Length", "constraint_bounds": [3, 12]},
    ]
    objectives = [max_stress_objective, max_displacement_objective, min_volume_objective]
    model_path = r"C:\Users\kevin\Documents\py-bayes-parametric\examples\FreeCADFEA\bracket_test.FCStd"
    opt = FreeCADFEMBayes(model_path, variables, objectives, n_init=10, n_iter=50)

    # Test case: Initialize and update with a predefined x
    predefined_x = [6, 6, 4, degree_to_radian(90), degree_to_radian(90), 4, 10]
    assert len(predefined_x) == len(variables)
    print("Testing update with predefined x:", predefined_x)
    results = opt.evaluate_objectives(predefined_x)
    print(results)

    # Run the optimization
    # X, Y = opt.optimize()
    # print("Optimization complete")
    
    # x_labels = [v['constraint_name'] for v in variables]
    # y_labels = ['stress', 'displacement', 'volume']
    # y_weights = [-1, -1, -1]
    # opt.output_npz(x_labels, y_labels, y_weights, 'data/bracket_optimization_results.npz')