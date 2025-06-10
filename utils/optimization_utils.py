import numpy as np

def opt_output_arrays_to_dict(X, Y, x_labels=None, y_labels=None, y_weights=None):
    """
    Convert optimization output arrays to a dictionary format.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input variables array of shape (N, nx) where N is number of samples and nx is number of inputs
    Y : numpy.ndarray
        Objective values array of shape (N, ny) where N is number of samples and ny is number of objectives
    x_labels : list of str, optional
        Labels for input variables. If None, will use 'x1', 'x2', etc.
    y_labels : list of str, optional
        Labels for objective functions. If None, will use 'y1', 'y2', etc.
    y_weights : list of float, optional
        Weights for objective functions. If None, will use equal weights.
    Returns
    -------
    dict
        Dictionary containing:
        - Input variables as separate arrays
        - Objective values as separate arrays
        - 'iteration' array for tracking sample order
    """
    N = X.shape[0]  # number of samples
    
    # Create default labels if none provided
    if x_labels is None:
        x_labels = [f'x{i+1}' for i in range(X.shape[1])]
    if y_labels is None:
        y_labels = [f'y{i+1}' for i in range(Y.shape[1])]
    
    # Initialize dictionary
    data_dict = {}
    
    # Add input variables
    for i, label in enumerate(x_labels):
        data_dict[label] = X[:, i]
    
    # Add objective values
    for i, label in enumerate(y_labels):
        data_dict[label] = Y[:, i] * y_weights[i]
    
    # Add iteration counter
    data_dict['iteration'] = np.arange(N)
    
    return data_dict, x_labels, y_labels

def save_optimization_dict(data_dict, x_labels, y_labels, filepath):
    """
    Save optimization dictionary and labels to an NPZ file.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing optimization data arrays
    x_labels : list of str
        Labels for input variables
    y_labels : list of str
        Labels for objective functions
    filepath : str
        Path where to save the NPZ file
    """
    # Create a new dictionary with all data
    save_dict = data_dict.copy()
    
    # Add labels as arrays
    save_dict['_x_labels'] = np.array(x_labels, dtype=str)
    save_dict['_y_labels'] = np.array(y_labels, dtype=str)
    
    # Save to NPZ file
    np.savez(filepath, **save_dict)
    print(f"Saved optimization data to {filepath}")