from typing import Union, Callable, Dict, Any, Tuple, List, Optional
import sympy as sp
import numpy as np
import math
import logging

class NumericalMethodBase:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.x = sp.Symbol('x')
        self.performance_metrics = {
            'start_time': 0,
            'end_time': 0,
            'elapsed_time': 0
        }

    def _start_timer(self):
        """Start performance timing."""
        import time
        self.performance_metrics['start_time'] = time.time()
        
    def _stop_timer(self):
        """Stop performance timing and calculate elapsed time."""
        import time
        self.performance_metrics['end_time'] = time.time()
        self.performance_metrics['elapsed_time'] = self.performance_metrics['end_time'] - self.performance_metrics['start_time']
        return self.performance_metrics['elapsed_time']
    
    def _get_elapsed_time(self):
        """Get the elapsed time in seconds."""
        return self.performance_metrics['elapsed_time']

    def _add_performance_data(self, table: List[Dict]) -> List[Dict]:
        """Add performance data to the results table."""
        if self.performance_metrics['elapsed_time'] > 0:
            # Check if performance data already exists in the table
            for row in table:
                if "Step" in row and row["Step"] == "Performance":
                    return table  # Performance data already exists, don't add it again
            
            # Add performance data if it doesn't already exist
            row = {
                "Step": "Performance",
                "Matrix" if "Matrix" in table[0] else "Value": f"{self.performance_metrics['elapsed_time']:.4f} seconds",
                "Operation" if "Operation" in table[0] else "Description": "Time taken to compute solution"
            }
            table.append(row)
        return table

    def _create_function(self, func_str: str) -> Callable:
        """
        Create a callable function from a string representation.
        
        Args:
            func_str: The function as a string (e.g., "x**2 - 4" or "sin(sqrt(x))")
            
        Returns:
            A callable function that can be evaluated with a numeric value
        """
        try:
            # Replace common math functions with sympy equivalents
            func_str = func_str.replace("math.sin", "sin")
            func_str = func_str.replace("math.cos", "cos")
            func_str = func_str.replace("math.tan", "tan")
            func_str = func_str.replace("math.log", "log")
            func_str = func_str.replace("math.log10", "log10")
            func_str = func_str.replace("math.exp", "exp")
            func_str = func_str.replace("math.sqrt", "sqrt")
            
            # Parse the function string into a sympy expression
            expr = sp.sympify(func_str)
            
            # Create a lambda function for faster evaluation
            f_lambda = sp.lambdify(self.x, expr, modules=['numpy', 'sympy'])
            
            # Create a callable function with comprehensive error handling
            def safe_eval(x):
                try:
                    # Suppress numpy warnings temporarily
                    with np.errstate(all='ignore'):
                        # Use the lambda function for faster evaluation
                        result = f_lambda(x)
                        
                        # Check for complex results (e.g., sqrt of negative numbers)
                        if isinstance(result, complex):
                            self.logger.warning(f"Complex result at x={x}")
                            return float('nan')
                            
                        # Check for NaN or infinity
                        if result is None or (hasattr(np, 'isnan') and np.isnan(result)) or (hasattr(np, 'isinf') and np.isinf(result)):
                            self.logger.warning(f"Invalid result at x={x}")
                            return float('nan')
                            
                        # Convert to float to ensure consistent return type
                        return float(result)
                except (ValueError, TypeError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
                    self.logger.warning(f"Function evaluation error at x={x}")
                    return float('nan')
            
            return safe_eval
        except Exception as e:
            self.logger.error(f"Error creating function from {func_str}")
            raise ValueError(f"Invalid function: {func_str}")

    def _create_derivative(self, func_str: str) -> Callable:
        """
        Create a callable derivative function from a string representation.
        
        Args:
            func_str: The function as a string (e.g., "x**2 - 4" or "sin(sqrt(x))")
            
        Returns:
            A callable function representing the derivative that can be evaluated with a numeric value
        """
        try:
            # Replace common math functions with sympy equivalents
            func_str = func_str.replace("math.sin", "sin")
            func_str = func_str.replace("math.cos", "cos")
            func_str = func_str.replace("math.tan", "tan")
            func_str = func_str.replace("math.log", "log")
            func_str = func_str.replace("math.log10", "log10")
            func_str = func_str.replace("math.exp", "exp")
            func_str = func_str.replace("math.sqrt", "sqrt")
            
            # Parse the function string into a sympy expression
            expr = sp.sympify(func_str)
            
            # Compute the derivative
            derivative = sp.diff(expr, self.x)
            
            # Create a lambda function for faster evaluation
            f_prime_lambda = sp.lambdify(self.x, derivative, modules=['numpy', 'sympy'])
            
            # Create a callable function with comprehensive error handling
            def safe_eval(x):
                try:
                    # Suppress numpy warnings temporarily
                    with np.errstate(all='ignore'):
                        # Use the lambda function for faster evaluation
                        result = f_prime_lambda(x)
                        
                        # Check for complex results (e.g., sqrt of negative numbers)
                        if isinstance(result, complex):
                            self.logger.warning(f"Complex derivative result at x={x}")
                            return float('nan')
                            
                        # Check for NaN or infinity
                        if result is None or (hasattr(np, 'isnan') and np.isnan(result)) or (hasattr(np, 'isinf') and np.isinf(result)):
                            self.logger.warning(f"Invalid derivative result at x={x}")
                            return float('nan')
                            
                        # Convert to float to ensure consistent return type
                        return float(result)
                except (ValueError, TypeError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
                    self.logger.warning(f"Derivative evaluation error at x={x}")
                    return float('nan')
            
            return safe_eval
        except Exception as e:
            self.logger.error(f"Error creating derivative from {func_str}")
            raise ValueError(f"Invalid function for derivative: {func_str}")

    def _round_value(self, value: Union[int, float], decimal_places: int) -> Union[int, float]:
        """
        Round a numeric value to the specified number of decimal places.
        
        This method implements an enhanced rounding algorithm that:
        1. Handles special cases (NaN, Inf)
        2. Uses banker's rounding for better numerical stability
        3. Handles very small values near zero appropriately
        4. Ensures consistent precision across all numerical methods
        5. Returns integers for whole numbers to avoid displaying trailing zeros
        
        Args:
            value: The number to round
            decimal_places: The number of decimal places to round to
            
        Returns:
            The rounded value as an integer if whole number, otherwise as a float
        """
        if not isinstance(value, (int, float)):
            return value
            
        # Handle special cases
        if math.isnan(value) or math.isinf(value):
            return value
            
        # Handle values very close to zero (avoid -0.0 results)
        if abs(value) < 1e-15:
            return 0
            
        # Special handling for very small values relative to decimal places
        if 0 < abs(value) < 10**(-decimal_places) and decimal_places > 0:
            # For very small values, use scientific notation internally
            # but keep the specified decimal places for display consistency
            magnitude = math.floor(math.log10(abs(value)))
            if magnitude < -decimal_places:
                # Round to significant figures instead of decimal places
                sig_figs = max(1, decimal_places - abs(magnitude) - 1)
                multiplier = 10 ** abs(magnitude+1)
                rounded = round(value * multiplier, sig_figs) / multiplier
                return rounded
        
        # For very large values, prevent unnecessary floating point errors
        if abs(value) > 1e10:
            precision = max(0, decimal_places - math.floor(math.log10(abs(value))))
            if precision <= 0:
                return round(value)
                
        # Standard case: regular decimal rounding
        multiplier = 10 ** decimal_places
        rounded = round(value * multiplier) / multiplier
        
        # Check if the rounded value is a whole number
        if rounded == int(rounded):
            return int(rounded)  # Return as integer if it's a whole number
        
        return rounded

    def _format_value(self, value: Union[int, float], decimal_places: int) -> str:
        """
        Format a numeric value as a string with the specified number of decimal places.
        
        This method formats the value for display purposes, removing trailing zeros
        after the decimal point if the number has no fractional part.
        
        Args:
            value: The number to format
            decimal_places: The number of decimal places to display
            
        Returns:
            A formatted string representation of the value
        """
        if not isinstance(value, (int, float)):
            return str(value)
            
        # Handle special cases
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        
        # For integers, return the string representation directly
        if isinstance(value, int) or value == int(value):
            return str(int(value))
            
        # Format with the specified precision
        formatted = f"{value:.{decimal_places}f}"
        
        # Remove trailing zeros and decimal point if needed
        if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.')
            
        return formatted

    def _format_error(self, error: Union[float, str], decimal_places: int) -> str:
        """
        Format an error value for display.
        
        This enhanced method:
        1. Handles special cases like first iteration "---"
        2. Ensures consistent display of error percentages
        3. Provides clearer representation for very small and very large errors
        4. Keeps the format appropriate for table display
        
        Args:
            error: The error value or "---" for first iteration
            decimal_places: Number of decimal places to round to
            
        Returns:
            A formatted string representation of the error
        """
        # Handle non-numeric error indicators
        if error == "---" or error is None:
            return "---"
            
        # Handle NaN and infinity cases
        if not isinstance(error, (int, float)) or math.isnan(error) or math.isinf(error):
            if math.isinf(error) if isinstance(error, float) else False:
                return "∞%" if error > 0 else "-∞%"
            return "NaN%"
        
        # Round the error value to appropriate precision
        # Use more precision for small errors to show meaningful digits
        if 0 < error < 0.001:
            # For very small errors, use scientific notation
            return f"{error:.2e}%"
        elif error > 10000:
            # For very large errors, use scientific notation
            return f"{error:.2e}%"
        else:
            # Use the specified decimal places for normal ranges
            digits = max(1, min(decimal_places, 6))  # Use between 1 and 6 digits
            rounded = round(error, digits)
            
            # Format with appropriate decimal places and remove trailing zeros
            if rounded == int(rounded):
                # For whole numbers, show no decimal places
                return f"{int(rounded)}%"
            else:
                # For fractional values, format with decimal places
                str_value = f"{rounded:.{digits}f}".rstrip('0').rstrip('.')
                return f"{str_value}%"

    def _check_convergence(self, error: float, eps: float, eps_operator: str) -> bool:
        """
        Check if the error satisfies the convergence criterion.
        
        Args:
            error: The current error value
            eps: The error tolerance
            eps_operator: The comparison operator ("<=", ">=", "<", ">", "=")
            
        Returns:
            True if the error satisfies the convergence criteria, False otherwise
        """
        try:
            # For each operator, return True when the stopping condition is met
            if eps_operator == "<=":
                return error <= eps  # Stop when error <= epsilon
            elif eps_operator == ">=":
                return error >= eps  # Stop when error >= epsilon
            elif eps_operator == "<":
                return error < eps   # Stop when error < epsilon
            elif eps_operator == ">":
                return error > eps   # Stop when error > epsilon
            elif eps_operator == "=":
                return abs(error - eps) < 1e-10  # Stop when error = epsilon (within tolerance)
            else:
                raise ValueError(f"Invalid epsilon operator: {eps_operator}")
        except Exception as e:
            self.logger.error(f"Error in convergence check")
            return False

    def solve(self, *args, **kwargs) -> Tuple[float, List[Dict]]:
        """
        Solve the numerical method. This method should be overridden by subclasses.
        
        Returns:
            A tuple containing the root and a list of dictionaries with iteration details
        """
        raise NotImplementedError("Subclasses must implement the solve method")