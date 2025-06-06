import subprocess
import os
import pandas as pd
import numpy as np
from io import StringIO

def naca_code(m, p, t):
    m_digit = int(round(m * 100))
    p_digit = int(round(p * 10))
    t_digits = int(round(t * 100))
    return f"NACA{m_digit}{p_digit}{t_digits:02d}"

def evaluate_naca_design(m, p, t, alpha, reynolds_number):
    code = naca_code(m, p, t)
    cl, cd = run_xfoil_single_alpha(naca_profile=code, reynolds_number=reynolds_number, alpha=alpha)
    return cl, cd

def evaluate_custom_design(dat_filepath: str, reynolds_number: int, alpha: float):
    cl, cd = run_xfoil_single_alpha(dat_filepath=dat_filepath, reynolds_number=reynolds_number, alpha=alpha)
    return cl, cd

def run_xfoil_single_alpha(dat_filepath: str=None, naca_profile: str=None, reynolds_number: int=3000000, alpha: float=5.0, output_filename: str="polar_output.txt", verbose: bool=False, timeout: int=5):
    """
    Run XFOIL for a single alpha angle.
    dat_filepath: str
    naca_profile: str
    reynolds_number: int
    alpha: float
    output_filename: str ## TODO: make this a path in this directory somehow
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

    print(f"Running XFOIL with commands: {load_foil_command}")
    xfoil_commands = f"""
{load_foil_command}
PANE
OPER
VISC {reynolds_number}
ITER
200
PACC
{output_filename}

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
        if verbose:
            print("XFOIL stdout:", result.stdout)
            print("XFOIL stderr:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"XFOIL failed with error code {e.returncode}")
        print("XFOIL stdout:", e.stdout)
        print("XFOIL stderr:", e.stderr)
        return np.nan, np.nan
    except subprocess.TimeoutExpired as e:
        print(f"XFOIL execution timed out after {timeout} seconds")
        return np.nan, np.nan
    except FileNotFoundError:
        print("XFOIL executable not found at the specified path")
        return np.nan, np.nan

    # Parse output
    output_filename = output_filename
    with open(output_filename, 'r') as f:
        lines = f.readlines()

    # Find the start of the data table
    data_start_line = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('alpha') and 'CL' in line and 'CD' in line:
            data_start_line = i + 1
            break
    if data_start_line == -1:
        raise ValueError("Could not find data table in XFOIL output.")

    # Read the data section using pandas.read_fwf for fixed-width columns
    # Only keep lines that look like data (start with a number or minus sign)
    data_lines = []
    for line in lines[data_start_line:]:
        if line.strip() == '' or not (line.strip()[0].isdigit() or line.strip()[0] == '-'):  # stop at next header or blank
            break
        data_lines.append(line)

    data_str = ''.join(data_lines)
    df = pd.read_fwf(StringIO(data_str),
                     names=['Alpha', 'Cl', 'Cd', 'Cdp', 'Cm', 'Top_Xtr', 'Bot_Xtr'])

    # Remove rows where 'Alpha' is not a number
    df = df[pd.to_numeric(df['Alpha'], errors='coerce').notnull()]

    # Convert columns to float
    for col in ['Alpha', 'Cl', 'Cd', 'Cdp', 'Cm', 'Top_Xtr', 'Bot_Xtr']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean up files
    os.remove(output_filename)

    # Return only the first valid Cl and Cd
    if not df.empty:
        cl_val = df['Cl'].iloc[0]
        cd_val = df['Cd'].iloc[0]
        return cl_val, cd_val
    else:
        return np.nan, np.nan

# --- Example Usage ---
if __name__ == "__main__":
    naca = "NACA2412"
    re = 3000000
    alpha = 5.0

    # Example usage of run_xfoil_single_alpha with NACA profile
    cl_single, cd_single = run_xfoil_single_alpha(naca_profile=naca, reynolds_number=re, alpha=alpha)
    print(f"\nSingle Alpha ({alpha} deg): Cl = {cl_single:.4f}, Cd = {cd_single:.6f}")

    # Example usage of run_xfoil_single_alpha with dat file
    dat_filepath = r"C:\Users\kevin\Documents\bayes-foil\data\airfoil.dat"  # TODO: make this work in the normal data directory
    cl_single, cd_single = run_xfoil_single_alpha(dat_filepath=dat_filepath, reynolds_number=re, alpha=alpha)
    print(f"\nSingle Alpha ({alpha} deg): Cl = {cl_single:.4f}, Cd = {cd_single:.6f}")