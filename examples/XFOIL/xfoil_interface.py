import subprocess
import os
import pandas as pd
import numpy as np
from io import StringIO
import re

def naca_code(m, p, t):
    m_digit = int(round(m * 100))
    p_digit = int(round(p * 10))
    t_digits = int(round(t * 100))
    return f"NACA{m_digit}{p_digit}{t_digits:02d}"

def evaluate_naca_design(m, p, t, alpha, RE):
    code = naca_code(m, p, t)
    cl, cd = run_xfoil_single_alpha(naca_profile=code, reynolds_number=RE, alpha=alpha)
    return cl, cd

def evaluate_custom_design(dat_filepath: str, RE: int, alpha: float):
    cl, cd = run_xfoil_single_alpha(dat_filepath=dat_filepath, reynolds_number=RE, alpha=alpha)
    return cl, cd

def run_xfoil_single_alpha(dat_filepath: str=None, 
                           naca_profile: str=None, 
                           reynolds_number: int=3000000, 
                           alpha: float=5.0, 
                           verbose: bool=False, 
                           timeout: int=5):
    """
    Run XFOIL for a single alpha angle.
    dat_filepath: str
    naca_profile: str
    reynolds_number: int
    alpha: float
    timeout: int - timeout in seconds for XFOIL execution
    
    returns: cl, cd
    """
    xfoil_path = r"C:/Users/kevin/Documents/bayes-foil/"
    if dat_filepath:
        load_foil_command = f"LOAD {dat_filepath}"
    elif naca_profile:
        load_foil_command = f"{naca_profile}"
    else:
        raise ValueError("Either dat_filepath or naca_profile must be provided")

    print(f"Running XFOIL with commands: {load_foil_command}, {reynolds_number}, {alpha}")
    xfoil_commands = f"""
{load_foil_command}
PANE
OPER
VISC {reynolds_number}
ITER
500
ALFA {alpha}

QUIT
"""
    xfoil_commands = xfoil_commands.strip()

    # Run XFOIL
    try:
        result = subprocess.run(
            [xfoil_path + "xfoil"],
            input=xfoil_commands,
            capture_output=True,
            text=True,
            check=True,  # This will raise CalledProcessError if the command fails
            timeout=timeout  # Add timeout parameter
        )
        stdout = result.stdout
        if verbose:
            print("XFOIL stdout:", stdout)
            print("XFOIL stderr:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"XFOIL failed with error code {e.returncode}")
        print("XFOIL stdout:", e.stdout)
        print("XFOIL stderr:", e.stderr)
        stdout = e.stdout
    except subprocess.TimeoutExpired as e:
        print(f"XFOIL execution timed out after {timeout} seconds")
        stdout = e.stdout if hasattr(e, 'stdout') and e.stdout is not None else ''
    except FileNotFoundError:
        print("XFOIL executable not found at the specified path")
        return np.nan, np.nan

    # Parse the LAST CL and CD from stdout
    cl_matches = re.findall(r"CL\s*=\s*([-+]?\d*\.\d+|\d+)", stdout)
    cd_matches = re.findall(r"CD\s*=\s*([-+]?\d*\.\d+|\d+)", stdout)
    if cl_matches and cd_matches:
        cl_val = float(cl_matches[-1])
        cd_val = float(cd_matches[-1])
        print(f"CL: {cl_val}, CD: {cd_val}")
        return cl_val, cd_val
    else:
        print("Could not find CL or CD in XFOIL stdout.")
        return np.nan, np.nan

# --- Example Usage ---
if __name__ == "__main__":
    naca = "NACA2412"
    RE = 3000000
    alpha = 3.0

    # Example usage of run_xfoil_single_alpha with NACA profile
    cl_single, cd_single = run_xfoil_single_alpha(naca_profile=naca, reynolds_number=RE, alpha=alpha, verbose=True)
    print(f"\nSingle Alpha ({alpha} deg): Cl = {cl_single:.4f}, Cd = {cd_single:.6f}")

    # Example usage of run_xfoil_single_alpha with dat file
    dat_filepath = r"C:\Users\kevin\Documents\bayes-foil\data\airfoil.dat"  # TODO: make this work in the normal data directory
    cl_single, cd_single = run_xfoil_single_alpha(dat_filepath=dat_filepath, reynolds_number=RE, alpha=alpha, verbose=True)
    print(f"\nSingle Alpha ({alpha} deg): Cl = {cl_single:.4f}, Cd = {cd_single:.6f}")