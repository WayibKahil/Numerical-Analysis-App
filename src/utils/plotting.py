import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, Union
import io
import tkinter as tk
from PIL import Image, ImageTk
import logging

class PlottingUtility:
    def __init__(self):
        """Initialize the plotting utility."""
        self.logger = logging.getLogger(__name__)
        
    def evaluate_function(self, func_str: str, x_values: np.ndarray) -> np.ndarray:
        """
        Safely evaluate a function string for an array of x values.
        
        Args:
            func_str: Function string (e.g., "x**2 - 4*x + 3")
            x_values: Array of x values
            
        Returns:
            Array of function values, with NaN for any errors
        """
        try:
            # Create safe evaluation environment
            import numpy as np
            from math import sin, cos, tan, exp, log, sqrt, pi
            
            # Create function
            def safe_eval(x):
                try:
                    # Replace common function names
                    expr = func_str.replace('math.', '')
                    
                    # Define the function environment
                    env = {
                        'x': x,
                        'np': np,
                        'sin': np.sin,
                        'cos': np.cos,
                        'tan': np.tan,
                        'exp': np.exp,
                        'log': np.log,
                        'sqrt': np.sqrt,
                        'pi': np.pi
                    }
                    
                    # Evaluate
                    return eval(expr, {"__builtins__": {}}, env)
                except Exception as e:
                    self.logger.error(f"Error evaluating function at x={x}: {str(e)}")
                    return np.nan
            
            # Apply to all x values
            y_values = np.array([safe_eval(x) for x in x_values])
            return y_values
            
        except Exception as e:
            self.logger.error(f"Error evaluating function: {str(e)}")
            return np.full_like(x_values, np.nan)

    def plot_function(self, func_str: str, x_range: Tuple[float, float] = None, 
                     roots: List[float] = None, iterations: List[Dict] = None,
                     title: str = "Function Plot", dpi: int = 100) -> Optional[ImageTk.PhotoImage]:
        """
        Plot a function with optional roots and iteration points.
        
        Args:
            func_str: Function string (e.g., "x**2 - 4*x + 3")
            x_range: Optional tuple (min_x, max_x) for x-axis range
            roots: List of root values to mark on the plot
            iterations: List of iteration dictionaries with 'Xi' values
            title: Plot title
            dpi: Resolution of the plot
            
        Returns:
            Tkinter PhotoImage object or None if error
        """
        try:
            # Create figure
            plt.figure(figsize=(8, 6), dpi=dpi)
            
            # Determine plot range
            if x_range is None:
                if roots and len(roots) > 0:
                    # Use roots to determine a reasonable range
                    root_values = [r for r in roots if isinstance(r, (int, float))]
                    if root_values:
                        mid_point = sum(root_values) / len(root_values)
                        max_dist = max(abs(r - mid_point) for r in root_values)
                        x_min = mid_point - max(max_dist * 2, 2)
                        x_max = mid_point + max(max_dist * 2, 2)
                    else:
                        x_min, x_max = -10, 10
                else:
                    x_min, x_max = -10, 10
            else:
                x_min, x_max = x_range
                
            # Generate x values
            x_values = np.linspace(x_min, x_max, 1000)
            
            # Evaluate function
            y_values = self.evaluate_function(func_str, x_values)
            
            # Find y range for a reasonable plot
            valid_indices = ~np.isnan(y_values)
            if np.any(valid_indices):
                valid_y = y_values[valid_indices]
                y_mean = np.mean(valid_y)
                y_std = np.std(valid_y)
                
                # Set reasonable y limits to avoid extreme values
                y_min = max(y_mean - 3 * y_std, min(valid_y))
                y_max = min(y_mean + 3 * y_std, max(valid_y))
                
                # Plot the function
                plt.plot(x_values, y_values, 'b-', label=f'f(x) = {func_str}')
                
                # Plot the x-axis
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                # Plot the roots
                if roots:
                    for i, root in enumerate(roots):
                        if isinstance(root, (int, float)):
                            root_y = self.evaluate_function(func_str, np.array([root]))[0]
                            plt.plot(root, root_y, 'ro', markersize=8, label=f'Root {i+1}: x = {root:.6g}')
                            plt.plot([root, root], [0, root_y], 'r--', alpha=0.5)
                
                # Plot the iterations
                if iterations:
                    x_values = []
                    for i, iteration in enumerate(iterations):
                        if 'Xi' in iteration and isinstance(iteration['Xi'], (int, float)):
                            x = float(iteration['Xi'])
                            x_values.append(x)
                            
                    if x_values:
                        y_values = self.evaluate_function(func_str, np.array(x_values))
                        plt.plot(x_values, y_values, 'go-', markersize=6, label='Iterations')
                
                # Set plot limits
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                
                # Add labels and legend
                plt.title(title)
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Convert plot to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=dpi)
                plt.close()
                
                # Convert to Tkinter PhotoImage
                buf.seek(0)
                img = Image.open(buf)
                return ImageTk.PhotoImage(img)
            else:
                plt.close()
                self.logger.warning("No valid function values to plot")
                return None
                
        except Exception as e:
            self.logger.error(f"Error plotting function: {str(e)}")
            if plt.get_fignums():
                plt.close()
            return None
            
    def plot_iteration_convergence(self, iterations: List[Dict], title: str = "Convergence Plot", 
                                  dpi: int = 100) -> Optional[ImageTk.PhotoImage]:
        """
        Plot the convergence of iteration values.
        
        Args:
            iterations: List of iteration dictionaries with 'Xi' values
            title: Plot title
            dpi: Resolution of the plot
            
        Returns:
            Tkinter PhotoImage object or None if error
        """
        try:
            # Create figure
            plt.figure(figsize=(8, 6), dpi=dpi)
            
            # Extract iteration values
            x_values = []
            error_values = []
            
            for i, iteration in enumerate(iterations):
                # Skip non-numeric or special message rows
                if (isinstance(iteration.get('Iteration'), int) or 
                    (isinstance(iteration.get('Iteration'), str) and 
                     iteration.get('Iteration').isdigit())):
                    
                    # Get x value
                    if 'Xi' in iteration and isinstance(iteration['Xi'], (int, float)):
                        x_values.append(float(iteration['Xi']))
                    elif 'x_i' in iteration and isinstance(iteration['x_i'], (int, float)):
                        x_values.append(float(iteration['x_i']))
                        
                    # Get error value
                    error_str = iteration.get('Error %', 'N/A')
                    if isinstance(error_str, str) and error_str not in ('N/A', '---'):
                        try:
                            # Remove % symbol if present
                            error_str = error_str.replace('%', '')
                            error_values.append(float(error_str))
                        except (ValueError, TypeError):
                            error_values.append(None)
                    elif isinstance(error_str, (int, float)):
                        error_values.append(float(error_str))
                    else:
                        error_values.append(None)
            
            # Plot x values convergence
            if x_values:
                plt.subplot(2, 1, 1)
                plt.plot(range(len(x_values)), x_values, 'bo-', markersize=6)
                plt.title(f"{title} - X Values")
                plt.xlabel('Iteration')
                plt.ylabel('X Value')
                plt.grid(True, alpha=0.3)
                
                # Plot error values if available
                valid_errors = [e for e in error_values if e is not None]
                if valid_errors:
                    plt.subplot(2, 1, 2)
                    plt.semilogy(range(len(valid_errors)), valid_errors, 'ro-', markersize=6)
                    plt.title("Error Convergence (log scale)")
                    plt.xlabel('Iteration')
                    plt.ylabel('Error')
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Convert plot to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=dpi)
                plt.close()
                
                # Convert to Tkinter PhotoImage
                buf.seek(0)
                img = Image.open(buf)
                return ImageTk.PhotoImage(img)
            else:
                plt.close()
                self.logger.warning("No valid iteration values to plot")
                return None
                
        except Exception as e:
            self.logger.error(f"Error plotting convergence: {str(e)}")
            if plt.get_fignums():
                plt.close()
            return None 