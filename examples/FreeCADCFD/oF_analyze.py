import pyvista as pv
import numpy as np
import os

def plane_average(mesh, scalar: str, normal: str = None, origin: tuple[float, float, float] = None) -> float:
    """
    Calculate the average value of a field over a plane.
    
    Args:
        mesh: The mesh to calculate the average value over.
        normal: The normal vector of the plane.
        origin: The origin of the plane.

    Returns:
        The average value of the field over the plane.
    """
    if normal is None and origin is None:
        return np.mean(mesh[scalar])
    else:
        plane = mesh.slice(normal=normal, origin=origin)
        return np.mean(plane[scalar])

def get_mesh(openfoam_case_path: str):
    """
    Get the mesh from the results.
    """
    reader = pv.POpenFOAMReader(openfoam_case_path)
    reader.CaseType = 'Decomposed Case'
    time_values = reader.time_values
    reader.set_active_time_value(time_values[-1])
    mesh = reader.read()

    internal_mesh = mesh['internalMesh']
    boundaries = mesh['boundary']
    return internal_mesh, boundaries
        
    
