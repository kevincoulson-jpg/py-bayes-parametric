import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import re
import ezdxf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from freecad_interface import FreeCADInterface


def generate_naca_4digit_airfoil(m, p, t, num_points=100, chord=1.0):
    """
    Generates the (x, y) coordinates for a NACA 4-digit airfoil.

    Parameters:
    m (float): Maximum camber as a fraction of the chord (e.g., 0.02 for 2%).
               Corresponds to the 1st digit M in NACA MPXX.
    p (float): Position of maximum camber as a fraction of the chord (e.g., 0.4 for 40%).
               Corresponds to the 2nd digit P in NACA MPXX.
    t (float): Maximum thickness as a fraction of the chord (e.g., 0.12 for 12%).
               Corresponds to the 3rd & 4th digits XX in NACA MPXX.
    num_points (int): Number of points along the chord for both upper and lower surfaces.
                      Points will be denser towards the leading and trailing edges.
    chord (float): The chord length of the airfoil (default is 1.0).

    Returns:
    tuple: (x, y) coordinates ordered for XFOIL compatibility
           x (np.array): x coordinates ordered from trailing edge (top) to trailing edge (bottom)
           y (np.array): y coordinates ordered from trailing edge (top) to trailing edge (bottom)
    """
    # Generate x-coordinates (cosine spacing for better distribution)
    # This places more points near leading and trailing edges
    x = np.cos(np.linspace(0, np.pi, num_points)) * -0.5 + 0.5
    x = x * chord  # Scale by chord length

    # Initialize arrays for y_c, dy_c_dx, and y_t
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    yt = np.zeros_like(x)

    # --- Calculate Mean Camber Line (yc) and its derivative (dyc_dx) ---
    if m == 0 or p == 0:
        yc[:] = 0.0
        dyc_dx[:] = 0.0
    else:
        for i, xi in enumerate(x):
            if 0 <= xi <= p * chord:
                yc[i] = (m / p**2) * (2 * p * xi - xi**2)
                dyc_dx[i] = (2 * m / p**2) * (p - xi)
            elif p * chord < xi <= chord:
                yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) * chord + 2 * p * xi - xi**2)
                dyc_dx[i] = (2 * m / (1 - p)**2) * (p - xi)

    # --- Calculate Thickness Distribution (yt) ---
    # Rescale x to be between 0 and 1 for the thickness formula
    x_norm = x / chord
    yt = 5 * t * (0.2969 * np.sqrt(x_norm) - 0.1260 * x_norm - 0.3516 * x_norm**2 + \
                  0.2843 * x_norm**3 - 0.1015 * x_norm**4)

    # --- Calculate Upper and Lower Surface Coordinates ---
    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Order points for XFOIL compatibility:
    # 1. Start at trailing edge (1.0, 0.0)
    # 2. Progress along upper surface to leading edge
    # 3. Progress along lower surface back to trailing edge
    # 4. End at trailing edge (1.0, 0.0)
    
    # Combine coordinates in the correct order
    x_ordered = np.concatenate([
        [1.0],           # Start at trailing edge
        xu[:-1][::-1],   # Upper surface from trailing to leading edge (excluding last point)
        xl[1:],          # Lower surface from leading to trailing edge (excluding first point)
        [1.0]            # End at trailing edge
    ])
    
    y_ordered = np.concatenate([
        [0.0],           # Start at trailing edge
        yu[:-1][::-1],   # Upper surface from trailing to leading edge (excluding last point)
        yl[1:],          # Lower surface from leading to trailing edge (excluding first point)
        [0.0]            # End at trailing edge
    ])

    return x_ordered, y_ordered

def format_airfoil_coordinates(x, y, output_file='C:/Users/kevin/Documents/py-bayes-parametric/examples/XFOIL/data/airfoil.dat'):
    """
    Format airfoil coordinates into the standard format for XFOIL.
    
    Parameters:
    -----------
    x, y : array-like
        Ordered coordinates (already in XFOIL-compatible order)
    output_file : str
        Path to output file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Open file for writing
    with open(output_file, 'w') as f:
        # Write header
        f.write('Custom Airfoil\n')
        
        # Write all points in order
        for xi, yi in zip(x, y):
            f.write(f'{xi:10.6f} {yi:10.6f}\n')


def de_boor(k, x, t, c, p):
    """
    Calculate a point on a B-spline using de Boor's algorithm.
    
    Parameters:
    -----------
    k : int
        Index of the knot interval containing x
    x : float
        Parameter value (0 to 1)
    t : array-like
        Knot vector
    c : array-like
        Control points
    p : int
        Degree of the spline
        
    Returns:
    --------
    array-like: Point on the spline
    """
    d = [c[j + k - p] for j in range(p + 1)]
    
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    
    return d[p]

def read_dxf_airfoil(dxf_file='C:/Users/kevin/Documents/py-bayes-parametric/examples/XFOIL/data/foil.dxf', num_points=75):
    """
    Read airfoil coordinates from a DXF file containing SPLINE entities.
    
    Parameters:
    -----------
    dxf_file : str
        Path to the DXF file
    num_points : int
        Number of points to sample along each spline
        
    Returns:
    --------
    tuple: (x_upper, y_upper, x_lower, y_lower) coordinates of the airfoil
    """
    try:
        # Read the DXF file
        doc = ezdxf.readfile(dxf_file)
        msp = doc.modelspace()
        
        # Get all SPLINE entities
        splines = msp.query('SPLINE')
        
        if not splines:
            raise ValueError("No splines found in DXF file")
            
        # We expect two splines: one for upper surface, one for lower
        if len(splines) != 2:
            raise ValueError(f"Expected 2 splines, found {len(splines)}")
            
        # Process each spline
        upper_points = []
        lower_points = []
        
        # Process each spline
        for i, spline in enumerate(splines):
            # Get control points
            control_points = np.array(spline.control_points)
            
            # Get knots
            knots = np.array(spline.knots)
            
            # Get degree
            degree = spline.dxf.degree
            
            # Sample points along the spline
            # Cosine spacing for better resolution at leading and trailing edges
            t = 0.5 * (1 - np.cos(np.linspace(0, np.pi, num_points)))
            points = []
            
            for ti in t:
                # Find the knot interval containing ti
                k = np.searchsorted(knots, ti) - 1
                k = max(degree, min(k, len(knots) - degree - 1))
                
                # Calculate point on spline using de Boor's algorithm
                point = de_boor(k, ti, knots, control_points, degree)
                points.append(point)
            
            # Store points for this spline
            if i == 0:  # First spline is upper surface
                upper_points = points
            else:  # Second spline is lower surface
                lower_points = points
        
        # Convert to numpy arrays
        x_upper = np.array([p[0] for p in upper_points])
        y_upper = np.array([p[1] for p in upper_points])
        x_lower = np.array([p[0] for p in lower_points])
        y_lower = np.array([p[1] for p in lower_points])
        
        # Normalize coordinates to chord length of 1.0
        x_min = min(np.min(x_upper), np.min(x_lower))
        x_max = max(np.max(x_upper), np.max(x_lower))
        scale = x_max - x_min
        
        x_upper = (x_upper - x_min) / scale
        y_upper = y_upper / scale
        x_lower = (x_lower - x_min) / scale
        y_lower = y_lower / scale
        
        return x_upper, y_upper, x_lower, y_lower
        
    except Exception as e:
        print(f"Error reading DXF file: {e}")
        return None, None, None, None

def organize_airfoil_coordinates(x_upper, y_upper, x_lower, y_lower):
    """
    Organize airfoil coordinates according to NACA format:
    1. Start at trailing edge (1.0, 0.0)
    2. Progress along upper surface to leading edge (0.0, 0.0)
    3. Progress along lower surface back to trailing edge (1.0, 0.0)
    
    Parameters:
    -----------
    x_upper, y_upper : array-like
        Upper surface coordinates
    x_lower, y_lower : array-like
        Lower surface coordinates
        
    Returns:
    --------
    tuple: (x, y) coordinates in NACA format
    """
    x = np.concatenate([x_lower[::-1], x_upper])
    y = np.concatenate([y_lower[::-1], y_upper])
    
    return x, y

# --- Example Usage and Visualization ---
if __name__ == "__main__":


    # Plot NACA airfoil
    naca_m = 2 / 100.0   # Max camber 2%
    naca_p = 4 / 10.0    # Max camber at 40% chord
    naca_xx = 12 / 100.0 # Max thickness 12%

    x, y = generate_naca_4digit_airfoil(naca_m, naca_p, naca_xx, num_points=150)

    plt.figure(figsize=(10, 3))
    plt.plot(x, y, 'b-', label='NACA Airfoil')
    plt.plot(x, y, 'r.', markersize=2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'NACA {int(naca_m*100)}{int(naca_p*10)}{int(naca_xx*100)} Airfoil')
    plt.xlabel('X/C')
    plt.ylabel('Y/C')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


    # Plot DXF airfoil

    interface = FreeCADInterface(output_json_path="C:/Users/kevin/Documents/py-bayes-parametric/examples/XFOIL/data")
    doc_path = "C:/Users/kevin/Documents/py-bayes-parametric/examples/XFOIL/customfoil.FCStd"
    print("Setting document:", doc_path)
    interface.set_document(doc_path)

    interface.export_dxf()
    print("DXF exported")

    x_upper, y_upper, x_lower, y_lower = read_dxf_airfoil(dxf_file="C:/Users/kevin/Documents/bayes-foil/data/foil.dxf") ## TODO: make this work in the normal data directory
    if x_upper is not None:
        # Plot raw coordinates
        plt.figure(figsize=(10, 3))
        plt.plot(x_upper, y_upper, 'g-', label='Upper Surface')
        plt.plot(x_lower, y_lower, 'm-', label='Lower Surface')
        plt.plot(x_upper, y_upper, 'k.', markersize=2)
        plt.plot(x_lower, y_lower, 'k.', markersize=2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('DXF Airfoil (Raw Coordinates)')
        plt.xlabel('X/C')
        plt.ylabel('Y/C')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
        
        # Plot organized coordinates
        x_ordered, y_ordered = organize_airfoil_coordinates(x_upper, y_upper, x_lower, y_lower)
        plt.figure(figsize=(10, 3))
        plt.plot(x_ordered, 'b-', label='Organized Airfoil X')
        plt.plot(y_ordered, 'r-', label='Organized Airfoil Y')
        plt.title('DXF Airfoil (Organized)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
    else:
        print("Error reading DXF file")

    # Format airfoil coordinates
    format_airfoil_coordinates(x_ordered, y_ordered, output_file="C:/Users/kevin/Documents/bayes-foil/data/airfoil.dat") ## TODO: make this work in the normal data directory
    print("Airfoil formatted, saved to C:/Users/kevin/Documents/bayes-foil/data/airfoil.dat")