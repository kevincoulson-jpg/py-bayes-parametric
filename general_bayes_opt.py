import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from utils.optimization_utils import opt_output_arrays_to_dict, save_optimization_dict

class GeneralBayesOpt:
    def __init__(self, bounds, evaluate_function, n_init=5, n_iter=20, device='cpu'):
        """
        bounds: list of (min, max) for each variable
        evaluate_function: callable that takes x (parameter values) and returns a list of objective values
        """
        print(f"[INIT GENERAL] Raw bounds: {bounds}")
        self.bounds = torch.tensor(bounds, dtype=torch.double, device=device)
        print(f"[INIT GENERAL] Tensor bounds: {self.bounds}")
        self.evaluate_function = evaluate_function
        self.n_init = n_init
        self.n_iter = n_iter
        self.device = device
        self.X = []
        self.Y = []

    def evaluate(self, x):
        """Evaluate all objectives at x (expects x as 1D numpy array)."""
        print(f"[EVAL GENERAL] x={x}")
        return self.evaluate_function(x)

    def initialize(self):
        for _, i in enumerate(range(self.n_init)):
            # Generate random values within bounds
            print("-"*20, f"[INIT] i={i}", "-"*20)
            x = np.random.uniform(self.bounds[0].cpu(), self.bounds[1].cpu())
            print(f"[INIT] x={x}")
            y = self.evaluate(x)
            self.X.append(x)
            self.Y.append(y)
            print(f"[INIT] y={y}")

    def optimize(self):
        self.initialize()
        X = torch.tensor(np.array(self.X), dtype=torch.double, device=self.device)
        Y = torch.tensor(np.array(self.Y), dtype=torch.double, device=self.device)
        for it in range(self.n_iter):
            if Y.shape[1] == 1:
                # Single-objective
                model = SingleTaskGP(X, Y)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                acqf = qExpectedImprovement(model, best_f=Y.max())
            else:
                # Multi-objective
                model = SingleTaskGP(X, Y)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                ref_point = Y.min(dim=0).values - 1e-3
                partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=Y)
                acqf = qExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=ref_point.tolist(),
                    partitioning=partitioning,
                    sampler=None  # Use default MC sampler
                )
            # Optimize acquisition function
            candidate, _ = optimize_acqf(
                acqf,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=100,
            )
            x_new = candidate.detach().cpu().numpy().flatten()
            print(f"[OPTIMIZE] New candidate: {x_new}")
            y_new = self.evaluate(x_new)
            X = torch.cat([X, candidate], dim=0)
            Y = torch.cat([Y, torch.tensor([y_new], dtype=torch.double, device=self.device)], dim=0)
            self.X.append(x_new)
            self.Y.append(y_new)
            print("-"*20, f"[ITER {it+1}] x={x_new}, y={y_new}", "-"*20)
        return np.array(self.X), np.array(self.Y)

    def output_npz(self, x_labels, y_labels, y_weights, filename):
        """
        Save the optimization results to an NPZ file using the utility functions.
        Args:
            x_labels (list of str): Labels for input variables
            y_labels (list of str): Labels for objective functions
            y_weights (list of float): Weights for objective functions
            filename (str): Path to save the NPZ file
        """
        X = np.array(self.X)
        Y = np.array(self.Y)
        data_dict, x_labels_out, y_labels_out = opt_output_arrays_to_dict(X, Y, x_labels, y_labels, y_weights)
        save_optimization_dict(data_dict, x_labels_out, y_labels_out, filename)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Single objective examples
    def single_obj_1d(x):
        return [np.sin(x[0]) + 0.1*x[0]**3]
    
    def single_obj_2d(x):
        return [np.sin(x[0]) + 0.1*x[0]**3 + np.cos(x[1]) + 0.2*x[1]**2]
    
    # Multi-objective examples
    def multi_obj_1d(x):
        return [np.sin(x[0]) + 0.1*x[0]**3, 
                np.cos(x[0]) + 0.2*x[0]**2]
    
    def multi_obj_2d(x):
        return [np.sin(x[0]) + 0.1*x[0]**3 + np.cos(x[1]) + 0.2*x[1]**2,
                np.cos(x[0]) + 0.2*x[0]**2 + np.sin(x[1]) + 0.1*x[1]**3]
    
    # 1D Single objective optimization
    bounds_1d = [[-5], [5]]
    single_opt_1d = GeneralBayesOpt(bounds_1d, single_obj_1d, n_init=3, n_iter=10)
    X_single_1d, Y_single_1d = single_opt_1d.optimize()
    
    # 2D Single objective optimization
    bounds_2d = [[-5, -5], [5, 5]]
    single_opt_2d = GeneralBayesOpt(bounds_2d, single_obj_2d, n_init=3, n_iter=10)
    X_single_2d, Y_single_2d = single_opt_2d.optimize()
    
    # 1D Multi-objective optimization
    multi_opt_1d = GeneralBayesOpt(bounds_1d, multi_obj_1d, n_init=3, n_iter=10)
    X_multi_1d, Y_multi_1d = multi_opt_1d.optimize()
    
    # 2D Multi-objective optimization
    multi_opt_2d = GeneralBayesOpt(bounds_2d, multi_obj_2d, n_init=3, n_iter=10)
    X_multi_2d, Y_multi_2d = multi_opt_2d.optimize()
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # 1D Single objective plot
    plt.subplot(221)
    x_plot = np.linspace(-5, 5, 100)
    plt.plot(x_plot, np.sin(x_plot) + 0.1*x_plot**3, 'b-', label='True function')
    plt.scatter(X_single_1d, Y_single_1d, c='r', label='Samples')
    plt.xlabel('x')
    plt.ylabel('f(x) = sin(x) + 0.1x³')
    plt.title('1D Single Objective')
    plt.legend()
    
    # 2D Single objective plot (showing best points)
    plt.subplot(222)
    plt.scatter(X_single_2d[:, 0], X_single_2d[:, 1], c=Y_single_2d, cmap='viridis')
    plt.colorbar(label='Objective value')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('2D Single Objective')
    
    # 1D Multi-objective plot
    plt.subplot(223)
    plt.scatter(Y_multi_1d[:, 0], Y_multi_1d[:, 1], c='r', label='Samples')
    plt.xlabel('f₁(x) = sin(x) + 0.1x³')
    plt.ylabel('f₂(x) = cos(x) + 0.2x²')
    plt.title('1D Multi-Objective')
    plt.legend()
    
    # 2D Multi-objective plot
    plt.subplot(224)
    plt.scatter(Y_multi_2d[:, 0], Y_multi_2d[:, 1], c='r', label='Samples')
    plt.xlabel('f₁(x) = sin(x₁) + 0.1x₁³ + cos(x₂) + 0.2x₂²')
    plt.ylabel('f₂(x) = cos(x₁) + 0.2x₁² + sin(x₂) + 0.1x₂³')
    plt.title('2D Multi-Objective')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

