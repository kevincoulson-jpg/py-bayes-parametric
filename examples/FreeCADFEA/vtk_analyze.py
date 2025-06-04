import pyvista as pv
import numpy as np
import os


## TODO: REFACTOR TO BE SIMILAR TO OF_ANALYZE.PY

def analyze_vtk_results(vtk_file_path: str):
    """
    Analyze VTK results from FreeCAD FEM analysis.
    vtk_file_path: path to the .vtu file containing the results
    """
    try:
        # Check if file exists
        if not os.path.exists(vtk_file_path):
            raise FileNotFoundError(f"VTK file not found at: {vtk_file_path}")

        # Read the VTK file
        mesh = pv.read(vtk_file_path)
        # print(f"Loaded mesh with {mesh.n_points} points and {mesh.n_cells} cells.")

        # Print available data arrays
        # print("\nAvailable point data arrays:", mesh.point_data.keys())
        # print("Available cell data arrays:", mesh.cell_data.keys())

        # Convert to UnstructuredGrid if it's not already
        if isinstance(mesh, pv.PolyData):
            mesh = mesh.cast_to_unstructured_grid()

        # Example analysis - get max von Mises stress
        if 'von Mises Stress' in mesh.point_data:
            von_mises = mesh.point_data['von Mises Stress']
            max_stress = np.max(von_mises)
            min_stress = np.min(von_mises)
            avg_stress = np.mean(von_mises)
            # print(f"\nVon Mises Stress Analysis:")
            # print(f"Max stress: {max_stress:.2f} MPa")
            # print(f"Min stress: {min_stress:.2f} MPa")
            # print(f"Average stress: {avg_stress:.2f} MPa")

        # Example analysis - get max displacement
        if 'Displacement Magnitude' in mesh.point_data:
            displacement = mesh.point_data['Displacement Magnitude']
            max_disp = np.max(displacement)
            # print(f"\nDisplacement Analysis:")
            # print(f"Max displacement: {max_disp:.6f} mm")

        return mesh

    except Exception as e:
        print(f"Error analyzing VTK results: {str(e)}")
        return None


## TODO: Fix this implementation
def analyze_point_data(mesh, point_coords: tuple):
    """
    Analyze data at a specific point in 3D space.
    mesh: PyVista mesh object
    point_coords: tuple of (x, y, z) coordinates
    """
    try:
        # Convert point coordinates to numpy array
        point = np.array(point_coords)
        
        # Find the cell containing the point
        cell_id = mesh.find_containing_cell(point)
        
        if cell_id != -1:
            print(f"\nAnalysis at point {point_coords}:")
            print(f"Point is inside cell {cell_id}")
            
            # Get the cell's point IDs
            cell = mesh.get_cell(cell_id)
            point_ids = cell.point_ids
            
            # Get the coordinates of the cell's points
            cell_points = mesh.points[point_ids]
            
            # Calculate barycentric coordinates for interpolation
            # This is a simplified approach - for more accurate results,
            # you might want to use proper FEM shape functions
            weights = np.ones(len(point_ids)) / len(point_ids)
            
            # Interpolate all point data
            for array_name in mesh.point_data.keys():
                values = mesh.point_data[array_name][point_ids]
                # Handle both scalar and vector data
                if values.ndim > 1:  # Vector data (like Displacement)
                    interpolated_value = np.sum(values * weights[:, np.newaxis], axis=0)
                else:  # Scalar data (like von Mises Stress)
                    interpolated_value = np.sum(values * weights)
                print(f"{array_name}: {interpolated_value}")
                
            return cell_id, point_ids
            
        else:
            print(f"\nPoint {point_coords} is outside the mesh")
            return None
            
    except Exception as e:
        print(f"Error analyzing point data: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    vtk_file = "C:/Users/kevin/Documents/FreeCAD-parametric-study/data/results.vtu"
    mesh = analyze_vtk_results(vtk_file)
    
    if mesh is not None:
        # Example: analyze data at a specific point
        # Replace these coordinates with your desired point
        point_to_analyze = (6.0, 0.0, 0.0)  # (x, y, z) coordinates
        point_data = analyze_point_data(mesh, point_to_analyze)