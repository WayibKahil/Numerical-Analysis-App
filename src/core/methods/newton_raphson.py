import sympy as sp
from .base import NumericalMethodBase
from typing import Tuple, List, Dict, Optional, Union, Any
import numpy as np
import math
from enum import Enum
import pandas as pd
from dataclasses import dataclass
from collections import OrderedDict

class ConvergenceStatus(str, Enum):
    """Enumeration for different convergence statuses."""
    CONVERGED = "converged"
    DIVERGED = "diverged"
    MAX_ITERATIONS = "max_iterations"
    ZERO_DERIVATIVE = "zero_derivative"
    NUMERICAL_ERROR = "numerical_error"
    DOMAIN_ERROR = "domain_error"
    OSCILLATING = "oscillating"
    ERROR = "error"

@dataclass
class NewtonRaphsonResult:
    """Data class to hold the result of a Newton-Raphson iteration."""
    root: Optional[float]
    iterations: List[Dict[str, Any]]
    status: ConvergenceStatus
    messages: List[str]
    iterations_table: pd.DataFrame
    
    def __iter__(self):
        """
        Allow the NewtonRaphsonResult to be unpacked as a tuple.
        This maintains backward compatibility with code that expects (root, iterations) tuple.
        
        Returns:
            Iterator that yields root and iterations
        """
        yield self.root
        yield self.iterations
        
    @classmethod
    def from_data(cls, root, iterations, status, messages, table_dict):
        """
        Create a NewtonRaphsonResult from raw data, with properly formatted DataFrame.
        
        Args:
            root: The estimated root (or None if not found)
            iterations: List of iteration details
            status: Convergence status
            messages: List of messages
            table_dict: OrderedDict containing iteration data
            
        Returns:
            NewtonRaphsonResult instance with properly formatted DataFrame
        """
        # Extract all iteration data into separate lists for each column
        iterations_data = []
        result_rows = []
        normal_rows = []
        
        # Collect rows by type
        for key, data in table_dict.items():
            # Skip rows that just contain NaN values 
            if key != "NaN" and not (isinstance(data.get("Iteration"), str) and data.get("Iteration") == "NaN"):
                # Create a dict for this row
                row = {
                    "Step": key
                }
                
                # Add all values from the data
                for col_name, value in data.items():
                    row[col_name] = value
                
                # Check if this is a result row
                is_result_row = (key == "Result" or key == "Final Result" or 
                                (isinstance(data.get("Iteration"), str) and data.get("Iteration") == "Result"))
                
                if is_result_row:
                    # Keep only the result row with the root value
                    result_rows.append(row)
                else:
                    normal_rows.append(row)
        
        # Only keep one result row (the last one added)
        if result_rows:
            # Get the last result row (which should be the most relevant one)
            final_result = result_rows[-1]
            final_result["highlight"] = True
            
            # Add normal rows first, then the final result row
            iterations_data = normal_rows + [final_result]
        else:
            iterations_data = normal_rows
        
        # Create a proper DataFrame
        if iterations_data:
            # Convert to DataFrame
            df = pd.DataFrame(iterations_data)
            
            # Keep only these specific columns in this order
            core_columns = [
                "Iteration", "Xi", "F(Xi)", "F'(Xi)", "Error%", "Xi+1"
            ]
            
            # Filter to only include columns that exist
            display_columns = [col for col in core_columns if col in df.columns]
                    
            # Reorder the DataFrame columns
            if display_columns:
                df = df[display_columns]
            
            # Fill NaN values with empty strings for cleaner display
            df = df.fillna("")
            
            # Remove any remaining NaN rows
            df = df[~(df['Iteration'].astype(str) == 'NaN')]
        else:
            df = pd.DataFrame()
            
        return cls(root, iterations, status, messages, df)

class NewtonRaphsonMethod(NumericalMethodBase):
    def __init__(self):
        """Initialize the Newton-Raphson method."""
        super().__init__()
        self.divergence_threshold = 1e15
        
    def _create_function(self, func_str: str):
        """
        Creates a callable function from a string with domain checking.
        
        Args:
            func_str: The function as a string
            
        Returns:
            A callable function with domain validation
        """
        import numpy as np
        
        # IMPORTANT: Using eval is a security risk if func_str comes from untrusted input
        allowed_names = {
            "np": np,
            "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "log10": np.log10,
            "abs": np.abs, "pi": np.pi, "e": np.e,
        }
        
        try:
            # Prepare the function string with domain checks
            safe_func_str = func_str
            
            # Check for sqrt to add domain validation
            if "sqrt" in safe_func_str:
                # Create a wrapper function that checks domain for sqrt
                func_code = f"""
def _user_func(x):
    # Handle scalar case
    if isinstance(x, (int, float)):
        # Domain check for sqrt
        if {"np.sqrt" in safe_func_str and "x < 0" or "sqrt(x)" in safe_func_str and "x < 0"}:
            return float('nan')  # Return NaN for invalid domain
        return {safe_func_str}
    else:
        # Handle array case (assume numpy array)
        import numpy as np
        result = np.full_like(x, np.nan, dtype=float)
        valid_mask = (x >= 0)  # Valid domain for sqrt
        result[valid_mask] = {safe_func_str.replace('x', 'x[valid_mask]')}
        return result
"""
            else:
                # Regular function without domain restrictions
                func_code = f"def _user_func(x): return {safe_func_str}"
            
            # Prepare the execution environment
            local_namespace = {}
            global_namespace = {"np": np}
            global_namespace.update(allowed_names)
            
            exec(func_code, global_namespace, local_namespace)
            return local_namespace['_user_func']
            
        except Exception as e:
            self.logger.error(f"Error creating function from '{func_str}': {e}")
            # Return a function that returns NaN to indicate error
            return lambda x: float('nan')

    def _create_derivative(self, func_str: str):
        """
        Creates a callable function for the derivative of a function string with domain checking.
        
        Args:
            func_str: The function as a string
            
        Returns:
            A callable function that evaluates the derivative with domain validation
        """
        import sympy as sp
        import numpy as np
        
        try:
            # Define symbolic variable
            x = sp.Symbol('x')
            
            # Fix common numpy functions before parsing
            preprocessed_func = func_str
            
            # Replace numpy functions with sympy equivalents
            replacements = {
                'np.sqrt': 'sqrt',
                'np.sin': 'sin',
                'np.cos': 'cos',
                'np.tan': 'tan',
                'np.exp': 'exp',
                'np.log': 'log'
            }
            
            for np_func, sp_func in replacements.items():
                preprocessed_func = preprocessed_func.replace(np_func, sp_func)
            
            # Replace ** with ^ for sympy
            preprocessed_func = preprocessed_func.replace('**', '^')
            
            # Parse the function using sympy
            try:
                f_sympy = sp.sympify(preprocessed_func)
            except Exception as parse_error:
                self.logger.error(f"Error parsing function: {parse_error}")
                raise ValueError(f"Could not parse function: {func_str}. Error: {parse_error}")
            
            # Calculate the derivative symbolically
            try:
                f_prime_sympy = sp.diff(f_sympy, x)
            except Exception as diff_error:
                self.logger.error(f"Error calculating derivative: {diff_error}")
                raise ValueError(f"Could not calculate derivative: {func_str}. Error: {diff_error}")
            
            # Convert back to a string with Python syntax
            f_prime_str = str(f_prime_sympy)
            
            # Replace sympy functions with numpy equivalents
            inverse_replacements = {
                'sqrt': 'np.sqrt',
                'sin': 'np.sin', 
                'cos': 'np.cos',
                'tan': 'np.tan',
                'exp': 'np.exp',
                'log': 'np.log'
            }
            
            for sp_func, np_func in inverse_replacements.items():
                f_prime_str = f_prime_str.replace(sp_func, np_func)
            
            # Replace ^ with ** for Python
            f_prime_str = f_prime_str.replace('^', '**')
            
            self.logger.debug(f"Original function: {func_str}")
            self.logger.debug(f"Calculated derivative: {f_prime_str}")
            
            # Create a domain-aware callable function
            # The derivative of a function with sqrt will have sqrt in denominator
            # which requires additional domain checking
            if "sqrt" in f_prime_str:
                code = f"""
def _derivative_func(x):
    # Handle scalar case
    if isinstance(x, (int, float)):
        # Domain check for sqrt and division by zero
        if x <= 0:  # Undefined at x=0 (division) and x<0 (sqrt)
            return float('nan')
        return {f_prime_str}
    else:
        # Handle array case (assume numpy array)
        import numpy as np
        result = np.full_like(x, np.nan, dtype=float)
        valid_mask = (x > 0)  # Valid domain for sqrt in denominator
        x_valid = x[valid_mask]
        try:
            result[valid_mask] = {f_prime_str.replace('x', 'x_valid')}
        except Exception as e:
            # Handle calculation errors
            pass  # Can't access self.logger from generated code
        return result
"""
                local_namespace = {}
                global_namespace = {"np": np}
                exec(code, global_namespace, local_namespace)
                return local_namespace['_derivative_func']
            else:
                # Use regular function creation for derivatives without domain issues
                return self._create_function(f_prime_str)
            
        except Exception as e:
            self.logger.error(f"Error creating derivative function: {str(e)}")
            # Return a function that returns NaN to indicate error
            return lambda x: float('nan')

    def solve(self, func_str: str, x0: float, eps: float = None, eps_operator: str = "<=", 
              max_iter: int = None, stop_by_eps: bool = True, decimal_places: int = 6,
              stop_criteria: str = "relative", consecutive_check: bool = False, 
              consecutive_tolerance: int = 3) -> NewtonRaphsonResult:
        """
        Solve for the root of a function using the Newton-Raphson method.
        
        The Newton-Raphson method uses the formula:
            x_{i+1} = x_i - f(x_i)/f'(x_i)
        
        This represents finding where the tangent line at the current point crosses the x-axis.
        The method converges quadratically when close to the root, making it very efficient.
        It works well with a wide range of functions including trigonometric (sin, cos, tan),
        exponential (exp), logarithmic (log), square roots (sqrt), and combinations thereof.
        
        Args:
            func_str: The function f(x) as a string (e.g., "-x**3 + 7.89*x + 11", "cos(x) - x", "exp(x) - 5")
            x0: Initial guess
            eps: Error tolerance (used if stop_by_eps is True)
            eps_operator: Comparison operator for epsilon check ("<=", ">=", "<", ">", "=")
            max_iter: Maximum number of iterations
            stop_by_eps: Whether to stop when error satisfies epsilon condition
            decimal_places: Number of decimal places for rounding
            stop_criteria: Stopping criteria type ("absolute", "relative", "function")
                - "absolute": Stop based on absolute error |x_{i+1} - x_i|
                - "relative": Stop based on approximate relative error |(x_{i+1} - x_i)/x_{i+1}| * 100%
                - "function": Stop based on function value |f(x_i)|
            consecutive_check: Whether to also check for convergence over consecutive iterations
            consecutive_tolerance: Number of consecutive iterations within tolerance to confirm convergence
            
        Returns:
            NewtonRaphsonResult containing the root, a list of dictionaries with iteration details,
            convergence status, messages, and a pandas DataFrame with the iterations table
        """
        # Ensure all required imports are available within the method
        import numpy as np
        
        # Validate inputs
        if not isinstance(func_str, str) or not func_str.strip():
            raise ValueError("Function string must be a non-empty string")
            
        try:
            x0 = float(x0)
        except (ValueError, TypeError):
            raise ValueError(f"Initial guess x0 must be convertible to float, got {x0}")
            
        # Use default values if not provided
        max_iter = max_iter if max_iter is not None else 50
        eps = eps if eps is not None else 0.0001
        
        # Validate other parameters
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError(f"Maximum iterations must be a positive integer, got {max_iter}")
            
        try:
            eps = float(eps)
            if eps <= 0:
                raise ValueError("Epsilon must be positive")
        except (ValueError, TypeError):
            raise ValueError(f"Epsilon must be convertible to float, got {eps}")
            
        if eps_operator not in ["<=", ">=", "<", ">", "="]:
            raise ValueError(f"Invalid epsilon operator: {eps_operator}")
            
        if stop_criteria not in ["absolute", "relative", "function"]:
            raise ValueError(f"Invalid stop criteria: {stop_criteria}")
            
        try:
            decimal_places = int(decimal_places)
            if decimal_places < 0:
                raise ValueError("Decimal places must be non-negative")
        except (ValueError, TypeError):
            raise ValueError(f"Decimal places must be an integer, got {decimal_places}")
            
        if not isinstance(consecutive_tolerance, int) or consecutive_tolerance <= 0:
            raise ValueError(f"Consecutive tolerance must be a positive integer, got {consecutive_tolerance}")
        
        # Initialize result table
        table = OrderedDict()
        
        try:
            self.logger.debug(f"Starting NewtonRaphsonMethod.solve with: func_str='{func_str}', x0={x0}, eps={eps}, max_iter={max_iter}")
            
            # Create function and its derivative
            try:
                f = self._create_function(func_str)
                f_prime = self._create_derivative(func_str)
            except Exception as e:
                self.logger.error(f"Failed to create function or derivative: {str(e)}")
                table["Initial Error"] = OrderedDict([
                    ("Iteration", "Error"),
                    ("Xi", "Function Creation"),
                    ("F(Xi)", str(e)),
                    ("F'(Xi)", "---"),
                    ("Error%", "---"),
                    ("Xi+1", "---")
                ])
                return NewtonRaphsonResult.from_data(None, [], ConvergenceStatus.ERROR, [f"Failed to create function or derivative: {str(e)}"], table)
            
            # Initialize variables
            x_current = float(x0)
            iter_count = 0
            relative_error = float('inf')
            error_display = "---"  # Initial error display
            consecutive_count = 0
            previous_values = []  # For cycle detection
            status = ConvergenceStatus.DIVERGED  # Default status
            
            # Check if initial guess is already a root
            try:
                fx_initial = float(f(x_current))
                if abs(fx_initial) < 1e-10:
                    self.logger.debug(f"Creating NewtonRaphsonResult with message: Initial guess is already a root")
                    table["Initial Check"] = OrderedDict([
                        ("Iteration", 0),
                        ("Xi", self._round_value(x_current, decimal_places)),
                        ("F(Xi)", "≈ 0"),
                        ("F'(Xi)", "---"),
                        ("Error%", "---"),
                        ("Xi+1", "---")
                    ])
                    
                    # Add result entry
                    table["Result"] = OrderedDict([
                        ("Iteration", "Result"),
                        ("Xi", self._round_value(x_current, decimal_places)),
                        ("F(Xi)", "Root found"),
                        ("F'(Xi)", "---"),
                        ("Error%", "---"),
                        ("Xi+1", self._round_value(x_current, decimal_places))
                    ])
                    
                    return NewtonRaphsonResult.from_data(x_current, [], ConvergenceStatus.CONVERGED, ["Initial guess is already a root (within numerical precision)"], table)
            except Exception as e:
                self.logger.error(f"Error evaluating function at initial guess: {str(e)}")
                table["Initial Error"] = OrderedDict([
                    ("Iteration", "Error"),
                    ("Xi", self._round_value(x_current, decimal_places)),
                    ("F(Xi)", f"Error evaluating function: {str(e)}"),
                    ("F'(Xi)", "---"),
                    ("Error%", "---"),
                    ("Xi+1", "---")
                ])
                return NewtonRaphsonResult.from_data(None, [], ConvergenceStatus.ERROR, [f"Error evaluating function at initial guess: {str(e)}"], table)
            
            # Main iteration loop
            for i in range(max_iter):
                iter_count = i
                x_old = x_current
                
                try:
                    # Step 1: Evaluate function and its derivative
                    fx = float(f(x_old))
                    fpx = float(f_prime(x_old))
                    
                    # Step 2: Check for zero or very small derivative (to avoid division by zero)
                    if abs(fpx) < 1e-10 or math.isnan(fpx):
                        status = ConvergenceStatus.ZERO_DERIVATIVE
                        # Add current iteration to table
                        table[f"Iteration {iter_count}"] = OrderedDict([
                            ("Iteration", iter_count),
                            ("Xi", self._round_value(x_old, decimal_places)),
                            ("F(Xi)", self._round_value(fx, decimal_places)),
                            ("F'(Xi)", "≈0" if abs(fpx) < 1e-10 else "NaN"),
                            ("Error%", error_display),
                            ("Xi+1", "---")
                        ])
                        # Add warning about zero derivative
                        table[f"Iteration {iter_count + 1}"] = OrderedDict([
                            ("Iteration", iter_count + 1),
                            ("Xi", "---"),
                            ("F(Xi)", "---"),
                            ("F'(Xi)", "---"),
                            ("Error%", "---"),
                            ("Xi+1", "---")
                        ])
                        return NewtonRaphsonResult.from_data(x_old, [], status, [f"{'Derivative is zero or very close to zero' if abs(fpx) < 1e-10 else 'Invalid derivative (NaN)'} at x = {self._round_value(x_old, decimal_places)}"], table)
                    
                    # Check if function value is NaN (domain error)
                    if math.isnan(fx):
                        status = ConvergenceStatus.DOMAIN_ERROR
                        # Add current iteration to table
                        table[f"Iteration {iter_count}"] = OrderedDict([
                            ("Iteration", iter_count),
                            ("Xi", self._round_value(x_old, decimal_places)),
                            ("F(Xi)", "NaN"),
                            ("F'(Xi)", self._round_value(fpx, decimal_places)),
                            ("Error%", error_display),
                            ("Xi+1", "---")
                        ])
                        # Add warning about domain error
                        table[f"Iteration {iter_count + 1}"] = OrderedDict([
                            ("Iteration", iter_count + 1),
                            ("Xi", "---"),
                            ("F(Xi)", "---"),
                            ("F'(Xi)", "---"),
                            ("Error%", "---"),
                            ("Xi+1", "---")
                        ])
                        return NewtonRaphsonResult.from_data(None, [], status, [f"Function value is invalid (NaN) at x = {self._round_value(x_old, decimal_places)}, likely outside domain"], table)
                    
                    # Step 3: Apply Newton-Raphson formula: x_{i+1} = x_i - f(x_i)/f'(x_i)
                    x_current = x_old - (fx / fpx)
                    
                    # Check for numerical issues (NaN, Inf)
                    if math.isnan(x_current) or math.isinf(x_current):
                        status = ConvergenceStatus.NUMERICAL_ERROR
                        # Add current iteration to table
                        table[f"Iteration {iter_count}"] = OrderedDict([
                            ("Iteration", iter_count),
                            ("Xi", self._round_value(x_old, decimal_places)),
                            ("F(Xi)", self._round_value(fx, decimal_places)),
                            ("F'(Xi)", self._round_value(fpx, decimal_places)),
                            ("Error%", error_display),
                            ("Xi+1", "NaN/Inf")
                        ])
                        # Add error message
                        table[f"Iteration {iter_count + 1}"] = OrderedDict([
                            ("Iteration", iter_count + 1),
                            ("Xi", "---"),
                            ("F(Xi)", "---"),
                            ("F'(Xi)", "---"),
                            ("Error%", "---"),
                            ("Xi+1", "---")
                        ])
                        return NewtonRaphsonResult.from_data(None, [], status, [f"Numerical error occurred at iteration {iter_count}"], table)
                    
                    # Step 4: Calculate absolute difference and relative error
                    abs_diff = abs(x_current - x_old)
                    
                    # Calculate error based on selected criteria
                    if abs(x_current) > 1e-15:
                        relative_error = (abs_diff / abs(x_current)) * 100  # percentage
                    elif abs_diff < 1e-15:
                        relative_error = 0.0  # both x_current and diff tiny, error is effectively zero
                    else:
                        relative_error = float('inf')  # x_current near zero but diff is significant
                    
                    # Format error for display - set to "---" for first iteration (i=0)
                    if i == 0:
                        error_display = "---"
                    else:
                        error_display = self._format_error(relative_error, decimal_places)
                    
                    # Choose error for convergence check based on stop_criteria
                    if stop_criteria == "absolute":
                        error_for_check = abs_diff
                        error_type_description = "absolute error"
                    elif stop_criteria == "relative":
                        error_for_check = relative_error
                        error_type_description = "relative error"
                    elif stop_criteria == "function":
                        error_for_check = abs(fx)
                        error_type_description = "function value"
                    else:
                        error_for_check = relative_error  # default to relative
                        error_type_description = "relative error"
                    
                    # Add iteration details to table with all key metrics for this iteration
                    table[f"Iteration {iter_count}"] = OrderedDict([
                        ("Iteration", iter_count),
                        ("Xi", self._round_value(x_old, decimal_places)),
                        ("F(Xi)", self._round_value(fx, decimal_places)),
                        ("F'(Xi)", self._round_value(fpx, decimal_places)),
                        ("Error%", error_display),
                        ("Xi+1", self._round_value(x_current, decimal_places))
                    ])
                    
                    # Step 5: Check if the function value is very close to zero (found exact root)
                    f_at_current = float(f(x_current))
                    if abs(f_at_current) < 1e-10:
                        status = ConvergenceStatus.CONVERGED
                        table[f"Iteration {iter_count + 1}"] = OrderedDict([
                            ("Iteration", iter_count + 1),
                            ("Xi", self._round_value(x_current, decimal_places)),
                            ("F(Xi)", "≈0"),
                            ("F'(Xi)", "---"),
                            ("Error%", "---"),
                            ("Xi+1", "---")
                        ])
                        
                        # Create a separate result row
                        table["Result"] = OrderedDict([
                            ("Iteration", "Result"),
                            ("Xi", self._round_value(x_current, decimal_places)),
                            ("F(Xi)", "≈0"),
                            ("F'(Xi)", "---"),
                            ("Error%", "---"),
                            ("Xi+1", self._round_value(x_current, decimal_places))
                        ])
                        
                        return NewtonRaphsonResult.from_data(x_current, [], status, ["Function value is zero within numerical precision"], table)
                    
                    # Step 6: Check for convergence based on error criteria
                    if i > 0 and stop_by_eps:
                        if self._check_convergence(error_for_check, eps, eps_operator):
                            if consecutive_check:
                                consecutive_count += 1
                                if consecutive_count >= consecutive_tolerance:
                                    status = ConvergenceStatus.CONVERGED
                                    stop_msg = f"Converged: {stop_criteria} error below {eps} for {consecutive_tolerance} consecutive iterations"
                                    table[f"Iteration {iter_count + 1}"] = OrderedDict([
                                        ("Iteration", iter_count + 1),
                                        ("Xi", self._round_value(x_current, decimal_places)),
                                        ("F(Xi)", "---"),
                                        ("F'(Xi)", "---"),
                                        ("Error%", "---"),
                                        ("Xi+1", "---")
                                    ])
                                    
                                    # Create a separate result row
                                    table["Result"] = OrderedDict([
                                        ("Iteration", "Result"),
                                        ("Xi", self._round_value(x_current, decimal_places)),
                                        ("F(Xi)", "---"),
                                        ("F'(Xi)", "---"),
                                        ("Error%", "---"),
                                        ("Xi+1", self._round_value(x_current, decimal_places))
                                    ])
                                    
                                    return NewtonRaphsonResult.from_data(x_current, [], status, [stop_msg], table)
                            else:
                                status = ConvergenceStatus.CONVERGED
                                stop_msg = f"Converged: {stop_criteria} error {eps_operator} {eps}"
                                table[f"Iteration {iter_count + 1}"] = OrderedDict([
                                    ("Iteration", iter_count + 1),
                                    ("Xi", self._round_value(x_current, decimal_places)),
                                    ("F(Xi)", "---"),
                                    ("F'(Xi)", "---"),
                                    ("Error%", "---"),
                                    ("Xi+1", "---")
                                ])
                                
                                # Create a separate result row
                                table["Result"] = OrderedDict([
                                    ("Iteration", "Result"),
                                    ("Xi", self._round_value(x_current, decimal_places)),
                                    ("F(Xi)", "---"),
                                    ("F'(Xi)", "---"),
                                    ("Error%", "---"),
                                    ("Xi+1", self._round_value(x_current, decimal_places))
                                ])
                                
                                return NewtonRaphsonResult.from_data(x_current, [], status, [stop_msg], table)
                        else:
                            consecutive_count = 0  # Reset if not meeting criteria
                    
                    # Step 7: Check for divergence (very large values)
                    if abs(x_current) > self.divergence_threshold:
                        status = ConvergenceStatus.DIVERGED
                        table[f"Iteration {iter_count + 1}"] = OrderedDict([
                            ("Iteration", iter_count + 1),
                            ("Xi", self._round_value(x_current, decimal_places)),
                            ("F(Xi)", "---"),
                            ("F'(Xi)", "---"),
                            ("Error%", "---"),
                            ("Xi+1", "---"),
                            ("Warning", "Method is diverging"),
                            ("Details", f"Root estimate exceeds safe bounds (|x| > {self.divergence_threshold:.2e})"),
                            ("Status", "DIVERGED")
                        ])
                        return NewtonRaphsonResult.from_data(None, [], status, [f"Method is diverging (value too large: {x_current:.2e})"], table)
                    
                    # Step 8: Check for oscillation/cycles
                    if len(previous_values) >= 2:
                        for prev_x in previous_values:
                            if abs(x_current - prev_x) < 1e-6:
                                table[f"Iteration {iter_count + 1}"] = OrderedDict([
                                    ("Iteration", iter_count + 1),
                                    ("Xi", self._round_value(x_current, decimal_places)),
                                    ("F(Xi)", "---"),
                                    ("F'(Xi)", "---"),
                                    ("Error%", "---"),
                                    ("Xi+1", "---"),
                                    ("Warning", "Method is oscillating between values"),
                                    ("Details", f"Consider using a different initial guess or method"),
                                    ("Status", "OSCILLATING")
                                ])
                                return NewtonRaphsonResult.from_data(x_current, [], ConvergenceStatus.OSCILLATING, [f"Method is oscillating between values. Stopping at iteration {iter_count}."], table)
                    
                    # Store value for cycle detection
                    previous_values.append(x_current)
                    if len(previous_values) > 5:  # Keep only the last 5 values
                        previous_values.pop(0)
                
                except Exception as e:
                    self.logger.error(f"Error in iteration {iter_count}: {str(e)}")
                    table[f"Iteration {iter_count}"] = OrderedDict([
                        ("Iteration", iter_count),
                        ("Xi", "---"),
                        ("F(Xi)", "---"),
                        ("F'(Xi)", "---"),
                        ("Error%", "---"),
                        ("Xi+1", "---")
                    ])
                    return NewtonRaphsonResult.from_data(None, [], ConvergenceStatus.ERROR, [f"Error in iteration {iter_count}: {str(e)}"], table)
            
            # If we reach here, max iterations were reached
            status = ConvergenceStatus.MAX_ITERATIONS
            table[f"Iteration {iter_count + 1}"] = OrderedDict([
                ("Iteration", iter_count + 1),
                ("Xi", self._round_value(x_current, decimal_places)),
                ("F(Xi)", "---"),
                ("F'(Xi)", "---"),
                ("Error%", "---"),
                ("Xi+1", "---")
            ])
            
            # Add a final row with the best approximation
            table["Result"] = OrderedDict([
                ("Iteration", "Result"),
                ("Xi", self._round_value(x_current, decimal_places)),
                ("F(Xi)", self._round_value(f(x_current), decimal_places)),
                ("F'(Xi)", "---"),
                ("Error%", error_display),
                ("Xi+1", self._round_value(x_current, decimal_places))
            ])
            
            return NewtonRaphsonResult.from_data(x_current, [], status, [f"Maximum iterations ({max_iter}) reached without convergence"], table)
            
        except Exception as e:
            # Handle any unexpected errors
            self.logger.error(f"Error in Newton-Raphson solve method: {str(e)}")
            table["Error"] = OrderedDict([
                ("Iteration", "Error"),
                ("Xi", "---"),
                ("F(Xi)", "---"),
                ("F'(Xi)", "---"),
                ("Error%", "---"),
                ("Xi+1", "---")
            ])
            return NewtonRaphsonResult.from_data(None, [], ConvergenceStatus.ERROR, [f"Newton-Raphson method failed: {str(e)}"], table)
            
    def _check_convergence(self, error_value: Union[float, str], eps: float, eps_operator: str) -> bool:
        """
        Check if the error satisfies the convergence criteria.
        
        Args:
            error_value: The error value (float or string)
            eps: Error tolerance
            eps_operator: Comparison operator for epsilon check ("<=", ">=", "<", ">", "=")
            
        Returns:
            bool: True if the error satisfies the convergence criteria, False otherwise
        """
        # Handle string error values or non-numeric values
        if not isinstance(error_value, (int, float)):
            return False
            
        # Convert error_value to float to ensure comparison works
        try:
            error_float = float(error_value)
        except (ValueError, TypeError):
            return False
            
        # Handle NaN and infinity
        if math.isnan(error_float) or math.isinf(error_float):
            return False
            
        # Ensure eps is a float
        try:
            eps_float = float(eps)
        except (ValueError, TypeError):
            return False
            
        self.logger.debug(f"Checking convergence: {error_float} {eps_operator} {eps_float}")
        
        # Perform the comparison
        try:
            if eps_operator == "<=":
                return error_float <= eps_float
            elif eps_operator == ">=":
                return error_float >= eps_float
            elif eps_operator == "<":
                return error_float < eps_float
            elif eps_operator == ">":
                return error_float > eps_float
            elif eps_operator == "=":
                return abs(error_float - eps_float) < 1e-9  # Tolerance for float equality
            else:
                self.logger.error(f"Invalid epsilon operator '{eps_operator}' in _check_convergence")
                return False
        except Exception as e:
            self.logger.exception(f"Error in convergence check: {str(e)}")
            return False
            
    def _round_value(self, value, decimal_places: int):
        """
        Rounds a value to the specified number of decimal places.
        Handles special cases like NaN and infinity.
        
        Args:
            value: The value to round
            decimal_places: Number of decimal places
            
        Returns:
            Rounded value or string representation for special cases
        """
        if not isinstance(value, (int, float)):
            return str(value)
            
        if math.isnan(value):
            return "NaN"
            
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
            
        return round(value, decimal_places)
        
    def _format_error(self, error, decimal_places: int) -> str:
        """
        Formats the error value, adding '%' for relative error.
        Handles special cases like NaN and infinity.
        
        Args:
            error: The error value
            decimal_places: Number of decimal places
            
        Returns:
            Formatted error string
        """
        if not isinstance(error, (int, float)):
            return str(error)
            
        if math.isnan(error):
            return "NaN"
            
        if math.isinf(error):
            return "Inf%" if error > 0 else "-Inf%"
            
        # For very small errors near zero
        if abs(error) < 1e-10:
            return "≈0%"
            
        return f"{error:.{decimal_places}f}%"