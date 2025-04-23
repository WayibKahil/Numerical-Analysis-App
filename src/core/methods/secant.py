from .base import NumericalMethodBase
from typing import Tuple, List, Dict, Optional, Union, Any, Callable
import numpy as np
from collections import OrderedDict
import pandas as pd
from enum import Enum, auto
from dataclasses import dataclass, field
import logging
import math
import time

class ConvergenceStatus(str, Enum):
    """Enumeration for different convergence statuses."""
    CONVERGED = "converged"
    DIVERGED = "diverged"
    MAX_ITERATIONS = "max_iterations"
    ROOT_FOUND = "root_found"
    BRACKETING_ERROR = "bracketing_error"
    DOMAIN_ERROR = "domain_error"
    EVALUATION_ERROR = "evaluation_error"
    COMPUTATION_ERROR = "computation_error"
    DERIVATIVE_ERROR = "derivative_error"
    STAGNATION = "stagnation"
    OSCILLATION = "oscillation"

@dataclass
class SecantResult:
    """Data class to hold the result of a Secant method iteration."""
    root: Optional[float] = None
    iterations: int = 0
    status: Optional[ConvergenceStatus] = None
    messages: List[str] = field(default_factory=list)
    iterations_table: pd.DataFrame = None
    execution_time: float = 0
    function_evaluations: int = 0
    convergence_rate: Optional[float] = None
            
    @classmethod
    def from_data(cls, root, iterations, status, messages, table_dict, 
                 execution_time=0, function_evaluations=0, convergence_rate=None):
        """
        Create a SecantResult from raw data, with properly formatted DataFrame.
        
        Args:
            root: The estimated root (or None if not found)
            iterations: Number of iterations performed
            status: Convergence status
            messages: List of messages
            table_dict: OrderedDict containing iteration data
            execution_time: Time taken to execute the method
            function_evaluations: Number of function evaluations performed
            convergence_rate: Estimated convergence rate if available
            
        Returns:
            SecantResult instance with properly formatted DataFrame
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
                    # Skip all unwanted columns
                    if (col_name != "Status" and col_name != "Rate" and 
                        col_name != "i" and col_name != "f(Xi-1)" and 
                        col_name != "f(Xi)" and col_name != "ea%"):
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
                "Iteration", "Xi-1", "F(Xi-1)", "Xi", "F(Xi)", "Xi+1", "Error%"
            ]
            
            # Filter to only include columns that exist
            display_columns = [col for col in core_columns if col in df.columns]
                    
            # Reorder the DataFrame columns
            if display_columns:
                df = df[display_columns]
            
            # Fill NaN values with empty strings for cleaner display
            df = df.fillna("")
            
            # Remove any remaining NaN rows
            if "Iteration" in df.columns:
                df = df[~(df['Iteration'].astype(str) == 'NaN')]
        else:
            df = pd.DataFrame()
            
        return cls(root, iterations, status, messages, df, 
                  execution_time, function_evaluations, convergence_rate)

class SecantMethod(NumericalMethodBase):
    """
    Implements the Secant method for finding roots of functions.
    
    The Secant method approximates the derivative in Newton-Raphson method
    using a finite difference. This makes it useful for functions whose
    derivatives are difficult to compute.
    
    Formula: x_{i+1} = x_i - [f(x_i)(x_{i-1} - x_i)] / [f(x_{i-1}) - f(x_i)]
    
    This method requires two initial guesses but does not require the function
    to change sign between these points (unlike bracketing methods).
    """
    
    def __init__(self):
        """Initialize the Secant method."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def solve(self, func_str: str, x0: float, x1: float, eps: float, eps_operator: str, 
              max_iter: int, stop_by_eps: bool, decimal_places: int = 6,
              stop_criteria: str = "absolute", consecutive_check: bool = False, 
              consecutive_tolerance: int = 3, aitken_acceleration: bool = False,
              detect_oscillations: bool = True, detect_stagnation: bool = True,
              stagnation_tolerance: float = 1e-12) -> SecantResult:
        """
        Solve for a root using the Secant method.
        
        Args:
            func_str: The function as a string (e.g., "0.95*x**3-5.9*x**2+10.9*x-6")
            x0: First initial guess (x_{i-1})
            x1: Second initial guess (x_i)
            eps: Error tolerance (for εa)
            eps_operator: Comparison operator for epsilon check (unused in this implementation)
            max_iter: Maximum number of iterations
            stop_by_eps: Whether to stop when error satisfies epsilon (unused in this implementation)
            decimal_places: Number of decimal places for rounding
            stop_criteria: Stopping criteria type (unused in this implementation)
            consecutive_check: Whether to check for convergence over consecutive iterations (unused)
            consecutive_tolerance: Number of consecutive iterations within tolerance (unused)
            aitken_acceleration: Whether to apply Aitken's acceleration (unused)
            detect_oscillations: Whether to detect oscillations (unused)
            detect_stagnation: Whether to detect stagnation (unused)
            stagnation_tolerance: Tolerance for stagnation detection (unused)
            
        Returns:
            SecantResult object with the solution details
        """
        # Start timer and track function evaluations
        start_time = time.time()
        function_evaluations = 0
        
        # Create result object
        result = SecantResult()
        
        # Initialize iteration table
        table_data = OrderedDict()
        
        # Create function from string
        try:
            f = self._create_function(func_str)
            
            # Initial values
            xi_minus_1 = float(x0)  # x_{i-1}
            xi = float(x1)          # x_i
            
            # Calculate function values at initial points
            fi_minus_1 = float(f(xi_minus_1))
            function_evaluations += 1
            fi = float(f(xi))
            function_evaluations += 1
            
            # Add initial values to table
            table_data["Iteration 0"] = OrderedDict([
                ("Iteration", 0),
                ("Xi-1", self._round_value(xi_minus_1, decimal_places)),
                ("F(Xi-1)", self._round_value(fi_minus_1, decimal_places)),
                ("Xi", self._round_value(xi, decimal_places)),
                ("F(Xi)", self._round_value(fi, decimal_places)),
                ("Xi+1", "---"),
                ("Error%", "---")
            ])
            
            # Main iteration loop
            for i in range(1, max_iter + 1):
                result.iterations = i
                
                # Check for division by zero
                if abs(fi - fi_minus_1) < 1e-10:
                    result.status = ConvergenceStatus.COMPUTATION_ERROR
                    result.messages.append("Division by zero in Secant computation.")
                    break
                
                # Calculate next approximation (implementing Equation 1.5)
                xi_plus_1 = xi - (fi * (xi_minus_1 - xi)) / (fi_minus_1 - fi)
                
                # Evaluate function at new point
                fi_plus_1 = float(f(xi_plus_1))
                function_evaluations += 1
                
                # Calculate approximate error (as percentage)
                if abs(xi_plus_1) > 1e-10:
                    ea = abs((xi_plus_1 - xi) / xi_plus_1) * 100
                else:
                    ea = abs(xi_plus_1 - xi) * 100
                    
                # Format error for display
                error_display = self._format_error(ea, decimal_places)
                
                # Add iteration to table
                table_data[f"Iteration {i}"] = OrderedDict([
                    ("Iteration", i),
                    ("Xi-1", self._round_value(xi_minus_1, decimal_places)),
                    ("F(Xi-1)", self._round_value(fi_minus_1, decimal_places)),
                    ("Xi", self._round_value(xi, decimal_places)),
                    ("F(Xi)", self._round_value(fi, decimal_places)),
                    ("Xi+1", self._round_value(xi_plus_1, decimal_places)),
                    ("Error%", error_display)
                ])
                
                # Check for convergence
                if ea <= eps:
                    result.root = xi_plus_1
                    result.status = ConvergenceStatus.CONVERGED
                    result.messages.append(f"Converged with error below tolerance (εa ≤ {eps}).")
                    break
                
                # Check for root (when function value is very close to zero)
                if abs(fi_plus_1) < 1e-10:
                    result.root = xi_plus_1
                    result.status = ConvergenceStatus.ROOT_FOUND
                    result.messages.append("Found exact root (f(x) ≈ 0).")
                    break
                
                # Update values for next iteration
                xi_minus_1, xi = xi, xi_plus_1
                fi_minus_1, fi = fi, fi_plus_1
            
            # Check if maximum iterations were reached
            if result.status is None:
                result.status = ConvergenceStatus.MAX_ITERATIONS
                result.messages.append(f"Maximum iterations ({max_iter}) reached.")
                result.root = xi_plus_1
            
            # Add result row
            if result.root is not None:
                table_data["Result"] = OrderedDict([
                    ("Iteration", "Result"),
                    ("Xi-1", "---"),
                    ("F(Xi-1)", "---"),
                    ("Xi", self._round_value(result.root, decimal_places)),
                    ("F(Xi)", self._round_value(f(result.root), decimal_places)),
                    ("Xi+1", "---"),
                    ("Error%", "---")
                ])
                
        except Exception as e:
            result.status = ConvergenceStatus.EVALUATION_ERROR
            result.messages.append(f"Error during computation: {str(e)}")
            
        # Record performance metrics
        result.execution_time = time.time() - start_time
        result.function_evaluations = function_evaluations
        
        # Return the formatted result
        return SecantResult.from_data(
            result.root, 
            result.iterations, 
            result.status, 
            result.messages, 
            table_data, 
            result.execution_time, 
            function_evaluations
        )
    
    def _format_error(self, error, decimal_places: int) -> str:
        """Format error value for display."""
        if error < 1e-10:
            return "≈0%"
        return f"{error:.{decimal_places}f}%"
        
    def solve_example_1_9(self) -> SecantResult:
        """
        Solve Example 1.9: f(x) = 0.95x³-5.9x²+10.9x-6
        
        Initial values: x₀=2.5, x₁=3.5
        Error tolerance: εa ≤ 0.5%
        
        Returns:
            SecantResult with the solution
        """
        return self.solve(
            func_str="0.95*x**3-5.9*x**2+10.9*x-6",
            x0=2.5,
            x1=3.5,
            eps=0.5,
            eps_operator="<=",
            max_iter=20,
            stop_by_eps=True,
            decimal_places=5
        )
        
    def solve_example_1_10(self) -> SecantResult:
        """
        Solve Example 1.10: f(x) = 2x³-11.7x²+17.7x-5
        
        Initial values: x₀=3, x₁=4
        Error tolerance: εa ≤ 0.5%
        
        Returns:
            SecantResult with the solution
        """
        return self.solve(
            func_str="2*x**3-11.7*x**2+17.7*x-5",
            x0=3,
            x1=4,
            eps=0.5,
            eps_operator="<=",
            max_iter=20,
            stop_by_eps=True,
            decimal_places=5
        )

    def solve_polynomial(self, coefficients: List[float], x0: float, x1: float, 
                        eps: float = 0.5, max_iter: int = 20, 
                        decimal_places: int = 6) -> SecantResult:
        """
        Specialized method to solve polynomial equations efficiently using the Secant method.
        
        Args:
            coefficients: List of polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
                          For example, [0.95, -5.9, 10.9, -6] for 0.95x³-5.9x²+10.9x-6
            x0: First initial guess
            x1: Second initial guess
            eps: Error tolerance
            max_iter: Maximum iterations
            decimal_places: Decimal places for display
            
        Returns:
            SecantResult object with the solution
        """
        # Create polynomial function string
        func_str = self._polynomial_to_string(coefficients)
        
        # Call the standard secant method with optimized parameters for polynomials
        return self.solve(
            func_str=func_str,
            x0=x0,
            x1=x1,
            eps=eps,
            eps_operator="<=",
            max_iter=max_iter,
            stop_by_eps=True,
            decimal_places=decimal_places,
            stop_criteria="absolute",
            consecutive_check=False,
            consecutive_tolerance=3,
            aitken_acceleration=False,
            detect_oscillations=True,
            detect_stagnation=True,
            stagnation_tolerance=1e-12
        )
        
    def _polynomial_to_string(self, coefficients: List[float]) -> str:
        """
        Convert polynomial coefficients to a function string.
        
        Args:
            coefficients: List of polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
            
        Returns:
            Function string representation of the polynomial
        """
        n = len(coefficients) - 1  # Highest degree
        terms = []
        
        for i, coef in enumerate(coefficients):
            if coef == 0:
                continue
                
            power = n - i
            if power == 0:
                terms.append(f"{coef}")
            elif power == 1:
                terms.append(f"{coef}*x")
            else:
                terms.append(f"{coef}*x**{power}")
                
        return "+".join(terms)
    
    def demo_specific_polynomial(self) -> SecantResult:
        """
        Demonstration method for the specific polynomial 0.95*x**3-5.9*x**2+10.9*x-6
        with initial values X-1=2.5, X0=3.5 and tolerance 0.5
        
        Returns:
            SecantResult with the solution
        """
        # Coefficients for 0.95*x**3-5.9*x**2+10.9*x-6
        coefficients = [0.95, -5.9, 10.9, -6]
        
        # Initial values
        x0 = 2.5  # X-1
        x1 = 3.5  # X0
        
        # Tolerance
        eps = 0.5
        
        # Solve using the specialized polynomial method
        result = self.solve_polynomial(
            coefficients=coefficients,
            x0=x0,
            x1=x1,
            eps=eps,
            max_iter=20,
            decimal_places=6
        )
        
        return result 

    # Fix all table entries to ensure consistency
    def _fix_table_entries(self, table_data: OrderedDict) -> OrderedDict:
        """
        Fix all table entries to ensure consistent columns and structure.
        
        Args:
            table_data: The table data to fix
            
        Returns:
            Fixed table data
        """
        fixed_table = OrderedDict()
        
        for key, entry in table_data.items():
            fixed_entry = OrderedDict()
            
            # Keep only necessary columns
            for col in ["Iteration", "Xi-1", "F(Xi-1)", "Xi", "F(Xi)", "Xi+1", "Error%"]:
                if col in entry:
                    fixed_entry[col] = entry[col]
                else:
                    fixed_entry[col] = "---"
            
            fixed_table[key] = fixed_entry
            
        return fixed_table
        
    def solve_polynomial_efficiently(self, a: float, b: float, c: float, d: float, 
                                   x0: float, x1: float, eps: float = 0.5, 
                                   max_iter: int = 20, decimal_places: int = 4) -> SecantResult:
        """
        Specialized method optimized specifically for cubic polynomials ax³+bx²+cx+d.
        
        Args:
            a, b, c, d: Coefficients of the cubic polynomial
            x0: First initial guess
            x1: Second initial guess
            eps: Error tolerance
            max_iter: Maximum iterations
            decimal_places: Decimal places for display
            
        Returns:
            SecantResult object with the solution
        """
        # Optimized cubic polynomial evaluation
        def f(x):
            return a*x**3 + b*x**2 + c*x + d
            
        # Start timer
        start_time = time.time()
        
        # Track function evaluations
        function_evaluations = 0
        
        # Create result object
        result = SecantResult()
        
        # Initialize iteration table
        table_data = OrderedDict()
        
        # Initial values
        xi_minus_1 = x0
        xi = x1
        
        # Calculate initial function values
        fi_minus_1 = f(xi_minus_1)
        function_evaluations += 1
        fi = f(xi)
        function_evaluations += 1
        
        # Add starting points to table
        table_data["Iteration 0"] = OrderedDict([
            ("Iteration", 0),
            ("Xi-1", self._round_value(xi_minus_1, decimal_places)),
            ("F(Xi-1)", self._round_value(fi_minus_1, decimal_places)),
            ("Xi", self._round_value(xi, decimal_places)),
            ("F(Xi)", self._round_value(fi, decimal_places)),
            ("Xi+1", "---"),
            ("Error%", "---")
        ])
        
        # Main loop
        for i in range(1, max_iter + 1):
            result.iterations = i
            
            # Check if denominator is close to zero
            if abs(fi - fi_minus_1) < 1e-10:
                result.status = ConvergenceStatus.COMPUTATION_ERROR
                result.messages.append("Division by zero in secant computation.")
                break
                
            # Calculate next approximation using secant formula
            xr = xi - fi * (xi - xi_minus_1) / (fi - fi_minus_1)
            
            # Calculate error
            ea = abs(xr - xi)
            error_display = self._format_error(ea, decimal_places)
            
            # Evaluate function at new point
            fr = f(xr)
            function_evaluations += 1
            
            # Add to table
            table_data[f"Iteration {i}"] = OrderedDict([
                ("Iteration", i),
                ("Xi-1", self._round_value(xi_minus_1, decimal_places)),
                ("F(Xi-1)", self._round_value(fi_minus_1, decimal_places)),
                ("Xi", self._round_value(xi, decimal_places)),
                ("F(Xi)", self._round_value(fi, decimal_places)),
                ("Xi+1", self._round_value(xr, decimal_places)),
                ("Error%", error_display)
            ])
            
            # Check for convergence
            if abs(fr) < 1e-10 or ea <= eps:
                result.root = xr
                result.status = ConvergenceStatus.CONVERGED
                result.messages.append(f"Converged with error: {ea}")
                break
                
            # Update for next iteration
            xi_minus_1, xi = xi, xr
            fi_minus_1, fi = fi, fr
        
        # Check if no convergence
        if result.status is None:
            result.status = ConvergenceStatus.MAX_ITERATIONS
            result.messages.append(f"Maximum iterations ({max_iter}) reached.")
            result.root = xr
        
        # Add result row
        table_data["Result"] = OrderedDict([
            ("Iteration", "Result"),
            ("Xi-1", "---"),
            ("F(Xi-1)", "---"),
            ("Xi", self._round_value(result.root, decimal_places)),
            ("F(Xi)", self._round_value(f(result.root), decimal_places)),
            ("Xi+1", "---"),
            ("Error%", "---")
        ])
        
        # Record performance metrics
        result.execution_time = time.time() - start_time
        result.function_evaluations = function_evaluations
        
        return SecantResult.from_data(result.root, result.iterations, result.status, 
                                     result.messages, table_data, result.execution_time, 
                                     function_evaluations)
                                     
    def solve_specific_polynomial(self) -> SecantResult:
        """
        Solves a specific polynomial with predefined coefficients and initial values.
        """
        # Define polynomial coefficients for: 0.95x³-5.9x²+10.9x-6
        coefficients = [0.95, -5.9, 10.9, -6]
        
        # Define initial guesses (these might need tuning for specific polynomials)
        x0 = 2.5
        x1 = 3.5
        
        # Set a reasonable tolerance
        eps = 0.5
        
        # Call the polynomial solver with our specific parameters
        result = self.solve_polynomial(
            coefficients=coefficients,
            x0=x0,
            x1=x1,
            eps=eps,
            max_iter=20,
            decimal_places=6
        )
        
        return result 