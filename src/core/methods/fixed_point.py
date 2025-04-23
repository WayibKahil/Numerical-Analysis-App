from .base import NumericalMethodBase
from typing import Tuple, List, Dict, Optional, Union, Callable
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
import math
from collections import OrderedDict

class StopCriteria(str, Enum):
    """Enumeration for different stopping criteria types."""
    RELATIVE = "relative"
    ABSOLUTE = "absolute"

class ConvergenceStatus(str, Enum):
    """Enumeration for different convergence statuses."""
    CONVERGED = "converged"
    DIVERGED = "diverged"
    MAX_ITERATIONS = "max_iterations"
    OSCILLATING = "oscillating"
    NUMERICAL_ERROR = "numerical_error"

@dataclass
class FixedPointResult:
    """Data class to hold the result of a fixed point iteration."""
    root: Optional[float]
    iterations: List[Dict]
    status: ConvergenceStatus
    message: str
    final_error: Optional[float]
    
    def __iter__(self):
        """
        Allow the FixedPointResult to be unpacked as a tuple.
        This maintains backward compatibility with code that expects (root, iterations) tuple.
        
        Returns:
            Iterator that yields root and iterations
        """
        yield self.root
        yield self.iterations

class FixedPointMethod(NumericalMethodBase):
    """
    Implements the Simple Fixed Point Iteration method to find roots of f(x) = 0.
    
    The fixed point iteration method finds a root by rearranging f(x) = 0 into x = g(x)
    and iterating x_{i+1} = g(x_i). The method converges if |g'(x)| < 1 near the root.
    
    Attributes:
        logger: Logger instance for logging messages
        max_history_size: Maximum number of previous values to store for cycle detection
        divergence_threshold: Threshold for detecting divergence
        cycle_tolerance: Tolerance for detecting cycles/oscillations
    """
    
    def __init__(self):
        """Initialize the FixedPointMethod with default parameters."""
        super().__init__()
        self.max_history_size = 5
        self.divergence_threshold = 1e15
        self.cycle_tolerance = 1e-7
        # Ensure logger is available, assuming it's set up in the base class
        if not hasattr(self, 'logger'):
             self.logger = logging.getLogger(__name__)
            # Ensure logging is configured in your main script or here
            # logging.basicConfig(level=logging.DEBUG) # Use DEBUG for more verbose output
        
    def _validate_inputs(self, f_str: Optional[str], g_str: Optional[str], x0: float, 
                        eps: float, max_iter: int, stop_criteria: str, auto_generate_g: bool = False) -> None:
        """
        Validate input parameters for the solve method.
        
        Args:
            f_str: The original function f(x) as a string
            g_str: The iteration function g(x) as a string
            x0: Initial guess
            eps: Error tolerance
            max_iter: Maximum number of iterations
            stop_criteria: Stopping criteria type
            auto_generate_g: Whether g(x) functions will be auto-generated
            
        Raises:
            ValueError: If any input parameter is invalid
        """
        if not f_str and not g_str:
            raise ValueError("Either f_str or g_str must be provided")
        
        # Allow f_str without g_str when auto_generate_g is True    
        if not g_str and not auto_generate_g and not f_str:
            raise ValueError("g_str must be provided when auto_generate_g is False and f_str is not provided")
            
        if not isinstance(x0, (int, float)):
            raise ValueError("Initial guess x0 must be a number")
            
        if eps <= 0:
            raise ValueError("Error tolerance eps must be positive")
            
        if max_iter <= 0:
            raise ValueError("Maximum iterations must be positive")
            
        if stop_criteria not in [criteria.value for criteria in StopCriteria]:
            raise ValueError(f"Invalid stop_criteria. Must be one of {[criteria.value for criteria in StopCriteria]}")
            
    def _check_convergence_condition(self, g_str: str, x0: float) -> bool:
        """
        Check if the convergence condition |g'(x)| < 1 is satisfied near the initial guess.
        
        Args:
            g_str: The iteration function g(x) as a string
            x0: Initial guess
            
        Returns:
            bool: True if convergence condition is satisfied, False otherwise
        """
        try:
            g_prime = self._create_derivative(g_str)
            g_prime_x0 = abs(g_prime(x0))
            return g_prime_x0 < 1
        except Exception:
            self.logger.warning("Could not verify convergence condition")
            return True  # Continue anyway, but log warning
            
    def solve(self, f_str: Optional[str], x0: float, eps: float = None, eps_operator: str = "<=", 
              max_iter: int = None, stop_by_eps: bool = True, decimal_places: int = 6,
              g_str: Optional[str] = None, 
              stop_criteria: str = StopCriteria.RELATIVE.value,
              consecutive_check: bool = False,
              consecutive_tolerance: int = 3,
              auto_generate_g: bool = False) -> Union[FixedPointResult, Tuple[Optional[float], List[Dict]]]:
        """
        Solve for a root of f(x) = 0 using the Fixed Point Iteration method.

        Finds a root by rearranging f(x) = 0 into x = g(x) and iterating x_{i+1} = g(x_i).
        If g(x) is not provided via g_str, it defaults to g(x) = f(x) + x.
        Convergence requires |g'(x)| < 1 near the root.

        Args:
            f_str: The original function f(x) as a string (e.g., "-0.9*x**2 + 1.7*x + 2.5").
                   Used if g_str is not provided to derive g(x) = f(x) + x.
            x0: Initial guess (xi).
            eps: Error tolerance (εa). Assumed to be in percentage if stop_criteria is 'relative'.
            eps_operator: Comparison operator for epsilon check ("<=", ">=", "<", ">", "=").
            max_iter: Maximum number of iterations.
            stop_by_eps: Whether to stop based on epsilon criteria.
            decimal_places: Number of decimal places for rounding output values.
            g_str: Optional. The specific iteration function g(x) as a string.
                   If None, g(x) is automatically derived as g(x) = f(x) + x.
            stop_criteria: Stopping criteria type ("relative", "absolute").
                - "relative": Stop based on approximate relative error |(x_{i+1} - x_i)/x_{i+1}| * 100%.
                - "absolute": Stop based on absolute error |x_{i+1} - x_i|.
            consecutive_check: Check for convergence over consecutive iterations.
            consecutive_tolerance: Number of consecutive iterations within tolerance.
            auto_generate_g: If True and g_str is None, automatically generate and select the best
                            g(x) function based on convergence criteria.

        Returns:
            Either a FixedPointResult, or for backward compatibility, a tuple containing (root, table)
            where root is the solution or None if not found, and table is a list of dictionaries
            containing iteration details.
        """
        # Use default values if not provided
        max_iter = max_iter if max_iter is not None else 50
        eps = eps if eps is not None else 0.0001
        
        # Initialize result table
        table = []
        
        try:
            self.logger.debug(f"Starting FixedPointMethod.solve with: f_str='{f_str}', x0={x0}, eps={eps}, max_iter={max_iter}, g_str='{g_str}', stop_criteria='{stop_criteria}'")
            
            # Validate inputs
            try:
                self._validate_inputs(f_str, g_str, x0, eps, max_iter, stop_criteria, auto_generate_g)
            except ValueError as e:
                self.logger.error(f"Input validation error: {e}")
                error_row = OrderedDict()
                error_row["Iteration"] = "Error"
                error_row["xi"] = "Validation"
                error_row["g(xi)"] = str(e)
                error_row["Error %"] = "---"
                return (None, [error_row])
            
            # --- Ensure eps is a float ---
            try:
                eps = float(eps)
            except (ValueError, TypeError) as e:
                err_msg = f"Invalid type for epsilon (eps): {eps}. Must be a number. Error: {e}"
                self.logger.error(err_msg)
                error_row = OrderedDict()
                error_row["Iteration"] = "Error"
                error_row["xi"] = "Epsilon"
                error_row["g(xi)"] = err_msg
                error_row["Error %"] = "---"
                return (None, [error_row])
            
            x_current = float(x0)
            consecutive_count = 0
            previous_values = []

            # Determine the iteration function g(x)
            if g_str:
                effective_g_str = g_str
                self.logger.info(f"Using user-provided iteration function g(x): {effective_g_str}")
            elif auto_generate_g and f_str:
                # Auto-generate g(x) functions and select the best one
                self.logger.info(f"Auto-generating g(x) functions from f(x) = {f_str}")
                
                # Generate candidate g(x) functions with convergence check
                candidate_gs = self.generate_g_functions(f_str=f_str, check_convergence=True, x_estimate=x0)
                
                # Log all candidates for informational purposes
                for i, candidate in enumerate(candidate_gs):
                    g_prime_info = f" |g'({x0})| = {candidate['g_prime_value']:.4f}" if candidate['g_prime_value'] is not None else ""
                    converges_info = f", likely {'converges' if candidate['converges'] else 'diverges'}" if candidate['converges'] is not None else ""
                    self.logger.info(f"Candidate {i+1}: g(x) = {candidate['g_str']}{g_prime_info}{converges_info}")
                
                # Filter for candidates that are likely to converge
                converging_candidates = [c for c in candidate_gs if c['converges'] is True]
                
                if converging_candidates:
                    # Sort by |g'(x0)| values - lower is better for convergence rate
                    converging_candidates.sort(key=lambda c: c['g_prime_value'])
                    best_candidate = converging_candidates[0]
                    effective_g_str = best_candidate['g_str']
                    
                    self.logger.info(f"Selected best g(x) function: {effective_g_str} with |g'({x0})| = {best_candidate['g_prime_value']:.4f}")
                    
                    # Add auto-generation info as regular table rows in the exact order requested
                    info_row1 = OrderedDict()
                    info_row1["Iteration"] = "Info"
                    info_row1["xi"] = "Auto-generation"
                    info_row1["g(xi)"] = f"Selected g(x) = {effective_g_str}"
                    info_row1["Error %"] = "---"
                    table.append(info_row1)
                    
                    info_row2 = OrderedDict()
                    info_row2["Iteration"] = "Info"
                    info_row2["xi"] = "Method"
                    info_row2["g(xi)"] = best_candidate['method']
                    info_row2["Error %"] = "---"
                    table.append(info_row2)
                    
                    info_row3 = OrderedDict()
                    info_row3["Iteration"] = "Info"
                    info_row3["xi"] = "Convergence"
                    info_row3["g(xi)"] = f"|g'({x0})| = {best_candidate['g_prime_value']:.4f} < 1"
                    info_row3["Error %"] = "---"
                    table.append(info_row3)
                else:
                    # If no converging candidates, use default
                    effective_g_str = f"({f_str}) + x"
                    self.logger.warning(f"No converging g(x) candidates found. Using default: g(x) = {effective_g_str}")
                    
                    # Add warnings as regular table rows
                    warning_row = OrderedDict()
                    warning_row["Iteration"] = "Warning"
                    warning_row["xi"] = "Auto-generation"
                    warning_row["g(xi)"] = "No g(x) functions likely to converge were found"
                    warning_row["Error %"] = "---"
                    table.append(warning_row)
                    
                    info_row = OrderedDict()
                    info_row["Iteration"] = "Info"
                    info_row["xi"] = "Default"
                    info_row["g(xi)"] = f"Using g(x) = {effective_g_str}"
                    info_row["Error %"] = "---"
                    table.append(info_row)
            else:
                effective_g_str = f"({f_str}) + x"
                self.logger.info(f"Deriving g(x) = f(x) + x from f(x) = {f_str}")
                
            # Check convergence condition
            if not self._check_convergence_condition(effective_g_str, x0):
                self.logger.warning("Convergence condition |g'(x)| < 1 may not be satisfied")
                
            # Create the iteration function
            try:
                g = self._create_function(effective_g_str)
            except Exception as e:
                err_msg = f"Failed to create function g(x) from string '{effective_g_str}': {e}"
                self.logger.error(err_msg)
                error_row = OrderedDict()
                error_row["Iteration"] = "Error"
                error_row["xi"] = "Function Creation"
                error_row["g(xi)"] = err_msg
                error_row["Error %"] = "---"
                return (None, [error_row])
            
            # Initialize error variables - ensure they are numeric where needed
            error_value_for_check = float('inf')  # Use infinity initially for numeric comparisons
            error_display = "---"  # String for display only, not for comparisons
            
            for i in range(max_iter):
                iteration_num = i # Start iteration count from 0 for table consistency with xi
                x_previous = x_current
                
                self.logger.debug(f"Iteration {iteration_num}: xi = {x_previous}")

                # Perform iteration: x_{i+1} = g(x_i)
                try:
                    x_next = float(g(x_previous))
                    self.logger.debug(f"Iteration {iteration_num}: g(xi) = x_next = {x_next}")
                    
                    # Check for numerical issues
                    if math.isnan(x_next) or math.isinf(x_next):
                        err_msg = f"Numerical instability (NaN/Inf) at iteration {iteration_num}. g({self._round_value(x_previous, 8)}) resulted in {x_next}."
                        self.logger.error(err_msg)
                        # Add current state as a regular row
                        current_row = OrderedDict()
                        current_row["Iteration"] = iteration_num
                        current_row["xi"] = self._round_value(x_previous, decimal_places)
                        current_row["g(xi)"] = "NaN/Inf" # Indicate failure
                        current_row["Error %"] = "---"
                        table.append(current_row)
                        
                        # Add error message as a row
                        error_row = OrderedDict()
                        error_row["Iteration"] = "Error"
                        error_row["xi"] = f"Iteration {iteration_num}"
                        error_row["g(xi)"] = err_msg
                        error_row["Error %"] = "---"
                        table.append(error_row)
                        return (None, table)
                        
                except Exception as e:
                    err_msg = f"Error evaluating g(x) at iteration {iteration_num} with xi = {self._round_value(x_previous, 8)}: {e}"
                    self.logger.exception(err_msg) # Log with traceback
                    
                    # Add current state as a regular row
                    current_row = OrderedDict()
                    current_row["Iteration"] = iteration_num
                    current_row["xi"] = self._round_value(x_previous, decimal_places)
                    current_row["g(xi)"] = "Eval Error"
                    current_row["Error %"] = "---"
                    table.append(current_row)
                    
                    # Add error message as a row
                    error_row = OrderedDict()
                    error_row["Iteration"] = "Error"
                    error_row["xi"] = f"Iteration {iteration_num}"
                    error_row["g(xi)"] = err_msg
                    error_row["Error %"] = "---"
                    table.append(error_row)
                    return (None, table)
                
                # --- Calculate Errors (Ensure results are numeric for checks) ---
                abs_diff = abs(x_next - x_previous)
                
                # Calculate relative error % numerically
                if abs(x_next) > 1e-15: # Increased tolerance slightly
                    rel_error_percent = (abs_diff / abs(x_next)) * 100.0
                elif abs_diff < 1e-15: # If both x_next and diff are tiny, error is effectively zero
                    rel_error_percent = 0.0
                else:
                    # x_next is near zero, but x_previous was different. Error is large.
                    rel_error_percent = float('inf')
                
                # Choose error for stopping check (guaranteed to be float or inf now)
                if stop_criteria == StopCriteria.RELATIVE.value:
                    error_value_for_check = rel_error_percent
                else: # stop_criteria == "absolute"
                    error_value_for_check = abs_diff
                
                # Format error for display (can be string like '---', 'inf%', 'nan%')
                error_display = self._format_error(rel_error_percent if i > 0 else "---", decimal_places) # Display '---' for first row i=0
                
                self.logger.debug(f"Iteration {iteration_num}: abs_diff={abs_diff:.4g}, rel_error%={rel_error_percent if isinstance(rel_error_percent, (int,float)) else rel_error_percent:.4g}, error_for_check={error_value_for_check:.4g}")
                
                # Create row for the iteration table
                row = OrderedDict()
                row["Iteration"] = iteration_num
                row["xi"] = self._round_value(x_current, decimal_places)
                row["g(xi)"] = self._round_value(x_next, decimal_places)
                row["Error %"] = error_display
                table.append(row)
                
                # --- Convergence Check (only after first iteration, i > 0) ---
                # Check only if i > 0 because error is calculated based on x_i and x_{i-1}
                converged = False
                stop_message = ""
                if i > 0 and stop_by_eps: # Check error criteria only from iteration 1 onwards
                    # Ensure error value is numeric and valid before convergence check
                    if isinstance(error_value_for_check, (int, float)) and not math.isnan(error_value_for_check) and not math.isinf(error_value_for_check):
                        self.logger.debug(f"Checking convergence: error={error_value_for_check}, eps={eps}, op='{eps_operator}'")
                        # Pass the numeric error_value_for_check to the check function
                        if self._check_convergence(error_value_for_check, eps, eps_operator):
                            if consecutive_check:
                                consecutive_count += 1
                                if consecutive_count >= consecutive_tolerance:
                                    converged = True
                                    stop_message = (f"Stopped: Converged below {eps} {stop_criteria} error "
                                                    f"for {consecutive_tolerance} consecutive iterations.")
                            else:
                                converged = True
                                stop_message = (f"Stopped: {stop_criteria.capitalize()} error "
                                                f"({error_value_for_check:.{decimal_places}f}) {eps_operator} {eps}.")
                        else:
                            consecutive_count = 0 # Reset count if not met
                    else:
                        self.logger.warning(f"Skipping convergence check for non-numeric error value: {error_value_for_check}")
                        consecutive_count = 0
                
                if converged:
                    self.logger.info(stop_message)
                    # Add the success message as a regular table row
                    result_row = OrderedDict()
                    result_row["Iteration"] = "Result"
                    result_row["xi"] = self._round_value(x_next, decimal_places)
                    result_row["g(xi)"] = stop_message
                    result_row["Error %"] = error_display
                    table.append(result_row)
                    return (self._round_value(x_next, decimal_places), table)
                
                # --- Check for Divergence ---
                if abs(x_next) > self.divergence_threshold:
                    warn_msg = f"Method may be diverging (value {x_next:.2e} at iter {iteration_num})."
                    self.logger.warning(warn_msg)
                    # Add the warning as a regular table row
                    warning_row = OrderedDict()
                    warning_row["Iteration"] = "Warning"
                    warning_row["xi"] = self._round_value(x_previous, decimal_places)
                    warning_row["g(xi)"] = warn_msg
                    warning_row["Error %"] = error_display
                    table.append(warning_row)
                    # Stop on divergence
                    return (None, table)
                
                # --- Check for Oscillation/Cycles ---
                cycle_detected = False
                if len(previous_values) >= 2:
                    # Simple 2-cycle check: x_next ~= x_before_previous
                    if abs(x_next - previous_values[-2]) < self.cycle_tolerance:
                        cycle_detected = True
                        warn_msg = (f"Potential oscillation detected around "
                                    f"{previous_values[-1]:.4f} and {x_next:.4f} at iter {iteration_num}.")
                
                if cycle_detected:
                    self.logger.warning(warn_msg)
                    # Add the warning as a regular table row
                    warning_row = OrderedDict()
                    warning_row["Iteration"] = "Warning"
                    warning_row["xi"] = self._round_value(x_next, decimal_places)
                    warning_row["g(xi)"] = warn_msg
                    warning_row["Error %"] = error_display
                    table.append(warning_row)
                    # Stop on oscillation (optional)
                    return (self._round_value(x_next, decimal_places), table)
                
                # Store values for cycle detection
                previous_values.append(x_next)
                if len(previous_values) > self.max_history_size:
                    previous_values.pop(0)
                
                # Prepare for next iteration
                x_current = x_next
            
            # Loop finished without meeting stopping criteria
            final_message = f"Stopped: Reached maximum iterations ({max_iter})."
            self.logger.info(final_message + f" Last estimate: {x_current}")
            # Add the final message as a regular table row
            result_row = OrderedDict()
            result_row["Iteration"] = "Result"
            result_row["xi"] = self._round_value(x_current, decimal_places)
            result_row["g(xi)"] = final_message
            result_row["Error %"] = error_display
            table.append(result_row)
            return (self._round_value(x_current, decimal_places), table)
            
        except Exception as e:
            self.logger.error(f"Error in solve method: {str(e)}")
            error_row = OrderedDict()
            error_row["Iteration"] = "Error"
            error_row["xi"] = "General"
            error_row["g(xi)"] = f"Error in solve method: {str(e)}"
            error_row["Error %"] = "---"
            return (None, [error_row])

    def _check_convergence(self, error_value: Union[float, str], eps: float, eps_operator: str) -> bool:
        """
        Check if the numeric error satisfies the convergence criteria.
        
        Args:
            error_value: The error value (float or string like '---' for first iteration)
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
            
        self.logger.debug(f"Performing comparison: {error_float} {eps_operator} {eps_float}")
        
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

    def generate_g_functions(self, f_str: str, check_convergence: bool = True, x_estimate: Optional[float] = None) -> List[Dict[str, Union[str, float, bool]]]:
        """
        Generate candidate g(x) functions for the fixed-point method, given f(x) = 0.
        
        Implements a systematic algorithm to generate different forms of g(x) by:
        1. The simple method: g(x) = f(x) + x
        2. Algebraic manipulation: isolating different terms containing x
        
        Args:
            f_str: The original function f(x) as a string (e.g., "-0.9*x**2 + 1.7*x + 2.5")
            check_convergence: Whether to check the convergence condition |g'(x)| < 1
            x_estimate: Estimated value near the root, used for convergence checking
            
        Returns:
            A list of dictionaries, each containing:
            - g_str: The generated g(x) function as a string
            - method: Description of the method used to generate this g(x)
            - converges: Boolean indicating if the function likely converges (if check_convergence=True)
            - g_prime_value: The value of |g'(x_estimate)| (if check_convergence=True and x_estimate provided)
        """
        import sympy as sp
        
        try:
            # Define symbolic variable
            x = sp.Symbol('x')
            
            # Parse the function using sympy
            f_sympy = sp.sympify(f_str.replace('np.', '').replace('**', '^').replace('sqrt', 'sqrt'))
            
            # Initialize list to store candidate g(x) functions
            candidates = []
            
            # Method 1: Simple addition - g(x) = f(x) + x
            g1_sympy = f_sympy + x
            g1_str = str(g1_sympy).replace('^', '**').replace('sqrt', 'np.sqrt')
            
            # Create the first candidate
            candidate1 = {
                "g_str": g1_str,
                "method": "Simple addition: g(x) = f(x) + x",
                "converges": None,
                "g_prime_value": None
            }
            
            candidates.append(candidate1)
            
            # Method 2: Try to find and isolate terms with x
            # First, expand the function to get a polynomial form
            f_expanded = sp.expand(f_sympy)
            
            try:
                # Try to collect terms by powers of x
                terms = sp.collect(f_expanded, x, evaluate=False)
                
                # Process linear term if it exists (coefficient of x^1)
                if x in terms:
                    coeff = terms[x]
                    if coeff != 0:
                        # Isolate the linear term: coeff*x = -(other terms)
                        other_terms = f_expanded - coeff*x
                        # Solve for x: x = -(other terms)/coeff
                        g_linear_sympy = -other_terms / coeff
                        g_linear_str = str(g_linear_sympy).replace('^', '**').replace('sqrt', 'np.sqrt')
                        
                        candidate_linear = {
                            "g_str": g_linear_str,
                            "method": f"Isolating linear term: x = -({other_terms})/({coeff})",
                            "converges": None,
                            "g_prime_value": None
                        }
                        
                        candidates.append(candidate_linear)
                
                # Process quadratic term if it exists (coefficient of x^2)
                if x**2 in terms:
                    coeff = terms[x**2]
                    if coeff != 0:
                        # Isolate the quadratic term: coeff*x^2 = -(other terms)
                        other_terms = f_expanded - coeff*(x**2)
                        # Solve for x: x = sqrt(-(other terms)/coeff)
                        # Need to handle both positive and negative coefficients
                        if sp.simplify(coeff) > 0:
                            g_quad_sympy = sp.sqrt(-other_terms / coeff)
                            g_quad_str = str(g_quad_sympy).replace('^', '**').replace('sqrt', 'np.sqrt')
                            method_desc = f"Isolating quadratic term (positive root): x = sqrt(-({other_terms})/({coeff}))"
                        else:
                            g_quad_sympy = sp.sqrt(other_terms / (-coeff))
                            g_quad_str = str(g_quad_sympy).replace('^', '**').replace('sqrt', 'np.sqrt')
                            method_desc = f"Isolating quadratic term (positive root): x = sqrt(({other_terms})/({-coeff}))"
                        
                        candidate_quad = {
                            "g_str": g_quad_str,
                            "method": method_desc,
                            "converges": None,
                            "g_prime_value": None
                        }
                        
                        candidates.append(candidate_quad)
                        
                        # Also add the negative root option
                        if sp.simplify(coeff) > 0:
                            g_quad_neg_sympy = -sp.sqrt(-other_terms / coeff)
                            g_quad_neg_str = str(g_quad_neg_sympy).replace('^', '**').replace('sqrt', 'np.sqrt')
                            method_desc = f"Isolating quadratic term (negative root): x = -sqrt(-({other_terms})/({coeff}))"
                        else:
                            g_quad_neg_sympy = -sp.sqrt(other_terms / (-coeff))
                            g_quad_neg_str = str(g_quad_neg_sympy).replace('^', '**').replace('sqrt', 'np.sqrt')
                            method_desc = f"Isolating quadratic term (negative root): x = -sqrt(({other_terms})/({-coeff}))"
                        
                        candidate_quad_neg = {
                            "g_str": g_quad_neg_str,
                            "method": method_desc,
                            "converges": None,
                            "g_prime_value": None
                        }
                        
                        candidates.append(candidate_quad_neg)
                
                # Handle other common forms like sin, cos, exp if they appear
                # This would require more complex pattern matching
            except Exception as e:
                self.logger.warning(f"Error in polynomial term isolation: {e}")
            
            # Check convergence of each candidate if requested
            if check_convergence and x_estimate is not None:
                for candidate in candidates:
                    try:
                        # Calculate g'(x) symbolically
                        g_sympy = sp.sympify(candidate["g_str"].replace('np.', '').replace('**', '^'))
                        g_prime_sympy = sp.diff(g_sympy, x)
                        
                        # Convert to Python function and evaluate at x_estimate
                        g_prime_func = sp.lambdify(x, g_prime_sympy, "numpy")
                        g_prime_value = abs(float(g_prime_func(x_estimate)))
                        
                        # Check if |g'(x_estimate)| < 1
                        candidate["converges"] = g_prime_value < 1
                        candidate["g_prime_value"] = g_prime_value
                        
                    except Exception as e:
                        self.logger.warning(f"Error checking convergence for {candidate['g_str']}: {e}")
                        # Keep None values for converges and g_prime_value
            
            # Return the list of candidates
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error generating g(x) functions: {e}")
            # Return a list with just the default g(x) = f(x) + x
            return [{
                "g_str": f"({f_str}) + x",
                "method": "Default method: g(x) = f(x) + x",
                "converges": None,
                "g_prime_value": None
            }]

    def _round_value(self, value: float, decimal_places: int) -> float:
        """Rounds a value to the specified number of decimal places."""
        if isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value):
            return round(value, decimal_places)
        return value  # Return as is if NaN, Inf, or other type

    def _format_error(self, error, decimal_places: int) -> str:
        """Formats the error value, adding '%' for relative error."""
        if isinstance(error, (int, float)) and not math.isnan(error) and not math.isinf(error):
            return f"{error:.{decimal_places}f}%"
        elif math.isinf(error) if isinstance(error, (int, float)) else False:
            return "inf%"
        elif math.isnan(error) if isinstance(error, (int, float)) else False:
            return "nan%"
        return str(error)  # Return as string if not numeric (e.g., '---', 'Div by ~0')

    # Assuming _create_function exists in base or is defined here
    def _create_function(self, func_str: str):
        """Creates a callable function from a string."""
        # IMPORTANT: Using eval is a security risk if func_str comes from untrusted input.
        # Consider safer alternatives like numexpr or sympy if needed.
        allowed_names = {
            "np": np,
            "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "log10": np.log10,
            "abs": np.abs, "pi": np.pi, "e": np.e,
            # Add other safe functions/constants as needed
        }
        # Prepare the function string
        func_code = f"def _user_func(x): return {func_str}"
        
        # Prepare the execution environment
        local_namespace = {}
        global_namespace = {"np": np} # Make numpy available
        global_namespace.update(allowed_names) # Add other allowed functions

        try:
            exec(func_code, global_namespace, local_namespace)
            return local_namespace['_user_func']
        except Exception as e:
            self.logger.error(f"Error compiling function string '{func_str}': {e}")
            raise ValueError(f"Invalid function string: {e}")

    def _create_derivative(self, func_str: str):
        """
        Creates a callable function for the derivative of a function string.
        
        Args:
            func_str: The function as a string
            
        Returns:
            A callable function that evaluates the derivative at a given point
        """
        import sympy as sp
        import numpy as np
        
        try:
            # Define symbolic variable
            x = sp.Symbol('x')
            
            # Parse the function using sympy, replacing numpy functions
            f_sympy = sp.sympify(func_str.replace('np.', '').replace('**', '^').replace('sqrt', 'sqrt'))
            
            # Calculate the derivative symbolically
            f_prime_sympy = sp.diff(f_sympy, x)
            
            # Convert back to Python function
            f_prime_str = str(f_prime_sympy).replace('^', '**').replace('sqrt', 'np.sqrt')
            
            # Use _create_function to create a callable
            return self._create_function(f_prime_str)
            
        except Exception as e:
            self.logger.error(f"Error creating derivative function: {str(e)}")
            # Create a simple fallback function that assumes non-convergence
            return lambda x: 2.0  # This will return |g'(x)| > 1, indicating potential non-convergence


# Example Usage (similar to the textbook examples):

if __name__ == '__main__':
    # Minimal test case for debugging
    solver = FixedPointMethod()
    print("Fixed Point Method initialized successfully")
    
    # Simple test function
    test = solver._create_function("x**2 - 4")
    print(f"Test function evaluation at x=2: {test(2)}")  # Should be 0


        