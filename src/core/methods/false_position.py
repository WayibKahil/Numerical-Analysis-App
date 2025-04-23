from .base import NumericalMethodBase
from typing import Tuple, List, Dict, Optional
import numpy as np
from collections import OrderedDict

class FalsePositionMethod(NumericalMethodBase):
    def solve(self, func_str: str, xl: float, xu: float, eps: float, eps_operator: str, max_iter: int, stop_by_eps: bool, decimal_places: int = 6,
          stop_criteria: str = "absolute", consecutive_check: bool = False, consecutive_tolerance: int = 3) -> Tuple[Optional[float], List[Dict]]:
        """
        Solve for the root of a function using the False Position method.
        
        The False Position method (Regula Falsi) works by approximating the function as a 
        straight line at each iteration. Instead of using the midpoint like Bisection,
        it computes a weighted midpoint based on function values.
        
        Key steps:
        1. Start with interval [xl, xu] where f(xl) and f(xu) have opposite signs
        2. Compute xr using linear interpolation: xr = xl - f(xl) * (xu - xl) / (f(xu) - f(xl))
        3. If f(xl) * f(xr) < 0, the root is in [xl, xr], so set xu = xr
        4. Otherwise, the root is in [xr, xu], so set xl = xr
        5. Repeat until convergence criteria are met
        
        Args:
            func_str: The function f(x) as a string (e.g., "x**2 - 4")
            xl: Lower bound of the interval
            xu: Upper bound of the interval
            eps: Stopping criterion threshold for the approximate relative error
            eps_operator: Comparison operator for error checking ('<', '<=', etc.)
            max_iter: Maximum number of iterations allowed
            stop_by_eps: Whether to stop by error threshold or max iterations
            decimal_places: Number of decimal places for rounding
            stop_criteria: Method to calculate error ("absolute" or "relative")
            consecutive_check: Whether to check for consecutive similar values
            consecutive_tolerance: Number of consecutive similar values for stopping
            
        Returns:
            Tuple containing the root and a list of dictionaries with iteration details
        """
        table = []
        iter_count = 0
        # Initialize xr and xr_old. Using xu is arbitrary, affects only first ea calc.
        xr = xu
        xr_old = xr
        ea = np.inf  # Approximate relative error (εa), initialized large
        
        # Use the appropriate stopping criteria based on parameters
        es = eps  # Use eps as our stopping criterion threshold
        imax = max_iter  # Use max_iter as our maximum iterations
        
        try:
            f = self._create_function(func_str)
            # Calculate initial function values ONCE (Optimized approach)
            fl = float(f(xl))
            fu = float(f(xu))
        except ValueError as e: # Error during function string parsing
             error_row = OrderedDict([("Error", str(e)), ("Status", "FUNCTION_ERROR")])
             return None, [error_row]
        except Exception as e: # Error during initial function evaluation
             error_row = OrderedDict([
                 ("Error", f"Initial function evaluation failed. Check function/bounds. Details: {e}"),
                 ("Status", "EVALUATION_ERROR")
             ])
             return None, [error_row]
        
        # --- Initial Checks ---
        root_tolerance = 1e-12 # Tolerance for checking if f(x) is zero

        # 1. Check if bounds are roots first
        if abs(fl) < root_tolerance:
            result_row = OrderedDict([
                ("Message", f"Initial lower bound {self._round_value(xl, decimal_places)} is already a root (f(xl) ≈ 0)."), ("Status", "SUCCESS"),
                ("Root", self._round_value(xl, decimal_places))
            ])
            return xl, [result_row]
        if abs(fu) < root_tolerance:
             result_row = OrderedDict([
                 ("Message", f"Initial upper bound {self._round_value(xu, decimal_places)} is already a root (f(xu) ≈ 0)."), ("Status", "SUCCESS"),
                 ("Root", self._round_value(xu, decimal_places))
             ])
             return xu, [result_row]

        # 2. Check bracketing condition: f(xl) * f(xu) < 0
        if fl * fu > 0:
            error_row = OrderedDict([
                ("Iteration", 0), 
                ("Xl", self._round_value(xl, decimal_places)), 
                ("f(Xl)", self._round_value(fl, decimal_places)),
                ("Xu", self._round_value(xu, decimal_places)), 
                ("f(Xu)", self._round_value(fu, decimal_places)),
                ("Error", "Initial interval does not bracket a root."),
                ("Status", "BRACKETING_ERROR"),
                ("Details", f"f(xl)={self._round_value(fl, decimal_places)} and f(xu)={self._round_value(fu, decimal_places)} must have opposite signs.")
            ])
            return None, [error_row]
        
        # --- Iteration Loop ---
        for i in range(imax):
            iter_count = i  # Start iteration count from 0 instead of 1
            xr_old = xr

            # Step 2: Calculate root estimate xr using False Position formula (Eq. 1.2)
            try:
                denominator = fl - fu
                if abs(denominator) < 1e-12: # Avoid division by near-zero
                    error_row = OrderedDict([
                        ("Iteration", iter_count), 
                        ("Xl", self._round_value(xl, decimal_places)), 
                        ("f(Xl)", self._round_value(fl, decimal_places)),
                        ("Xu", self._round_value(xu, decimal_places)), 
                        ("f(Xu)", self._round_value(fu, decimal_places)),
                        ("Error", "Denominator f(xl) - f(xu) is near zero. Cannot compute Xr reliably."),
                        ("Status", "COMPUTATION_ERROR")
                    ])
                    table.append(error_row)
                    # Return the previous best estimate as the calculation failed
                    return xr_old, table 

                # Use Eq 1.2 from the textbook
                xr = xu - fu * (xl - xu) / denominator

            except ZeroDivisionError: # Should be caught by the near-zero check, but just in case
                 error_row = OrderedDict([
                     ("Iteration", iter_count), 
                     ("Xl", self._round_value(xl, decimal_places)), 
                     ("f(Xl)", self._round_value(fl, decimal_places)),
                     ("Xu", self._round_value(xu, decimal_places)), 
                     ("f(Xu)", self._round_value(fu, decimal_places)),
                     ("Error", "Division by zero: f(xl) - f(xu) = 0. Cannot compute Xr."),
                     ("Status", "COMPUTATION_ERROR")
                 ])
                 table.append(error_row)
                 return xr_old, table # Return previous estimate

            # Calculate f(xr) - Only one function evaluation needed here per iteration
            try:
                fr = float(f(xr))
            except Exception as e: # Handle evaluation errors at xr
                 error_row = OrderedDict([
                     ("Iteration", iter_count), 
                     ("Xl", self._round_value(xl, decimal_places)), 
                     ("f(Xl)", self._round_value(fl, decimal_places)),
                     ("Xu", self._round_value(xu, decimal_places)), 
                     ("f(Xu)", self._round_value(fu, decimal_places)),
                     ("Xr", self._round_value(xr, decimal_places)),
                     ("Error", f"Function evaluation failed at Xr={self._round_value(xr, decimal_places)}. Details: {e}"), 
                     ("Status", "EVALUATION_ERROR")
                 ])
                 table.append(error_row)
                 return None, table # Cannot proceed

            # Calculate approximate relative error (ea) %
            # Calculate AFTER the first iteration (when xr_old is meaningful)
            if iter_count > 0:
                 if abs(xr) > 1e-12: # Avoid division by zero/very small xr
                     ea = abs((xr - xr_old) / xr) * 100
                 else:
                     # If xr is very close to zero, relative error is unstable/undefined.
                     # Report '---' or use absolute difference (less standard for this method's stopping criterion)
                     ea = "---" # Indicate undefined relative error
            else: # First iteration
                 ea = "---" 

            # --- Populate Table Row (matching book examples) ---
            row = OrderedDict([
                ("Iteration", iter_count),
                ("Xl", self._round_value(xl, decimal_places)),
                ("f(Xl)", self._round_value(fl, decimal_places)),
                ("Xu", self._round_value(xu, decimal_places)),
                ("f(Xu)", self._round_value(fu, decimal_places)),
                ("Xr", self._round_value(xr, decimal_places)),
                ("f(Xr)", self._round_value(fr, decimal_places)),
                ("Error %", self._format_error(ea, decimal_places) if iter_count > 0 else "---")
            ])

            # --- Check Termination Conditions ---
            # 1. Check for exact root (f(xr) close to zero)
            if abs(fr) < root_tolerance:
                ea = 0.0 # Error is effectively zero
                row["Error %"] = self._format_error(ea, decimal_places) # Update final row error
                row["Status"] = "EXACT_ROOT" # Match bisection method status
                table.append(row)
                
                # Add highlighted result row - match bisection format exactly
                final_result_row = OrderedDict([
                    ("Iteration", "Result"),
                    ("Xl", ""),
                    ("f(Xl)", ""),
                    ("Xu", ""),
                    ("f(Xu)", ""),
                    ("Xr", self._round_value(xr, decimal_places)),
                    ("f(Xr)", self._round_value(fr, decimal_places)),
                    ("Error %", "0%")
                ])
                table.append(final_result_row)
                
                result_row = OrderedDict([
                    ("Message", f"Exact root found."),
                    ("Status", "SUCCESS"),
                    ("Details", f"f(Xr) = 0 at Xr = {self._round_value(xr, decimal_places)} in iteration {iter_count}."),
                    ("Root", self._round_value(xr, decimal_places))
                ])
                table.append(result_row)
                return xr, table

            # 2. Check approximate relative error against stopping criterion 'es'
            stop_by_error = False
            # Only check if 'ea' is a valid number (not "---") and if stop_by_eps is True
            if isinstance(ea, (int, float)) and stop_by_eps: 
                # Use the proper comparison based on eps_operator
                should_stop = self._check_convergence(ea, es, eps_operator)
                if should_stop:
                    stop_by_error = True
                    row["Status"] = "CONVERGED" # Match bisection method status
                    table.append(row) # Append the final successful iteration row
                    
                    # Add highlighted result row - match bisection format exactly
                    final_result_row = OrderedDict([
                        ("Iteration", "Result"),
                        ("Xl", ""),
                        ("f(Xl)", ""),
                        ("Xu", ""),
                        ("f(Xu)", ""),
                        ("Xr", self._round_value(xr, decimal_places)),
                        ("f(Xr)", self._round_value(fr, decimal_places)),
                        ("Error %", self._format_error(ea, decimal_places))
                    ])
                    table.append(final_result_row)
                    
                    result_row = OrderedDict([
                        ("Message", f"Converged successfully."),
                        ("Status", "CONVERGED"),
                        ("Details", f"Stopped because εa ({self._format_error(ea, decimal_places)}) {eps_operator} es ({es}%) at iteration {iter_count}."),
                        ("Root", self._round_value(xr, decimal_places))
                    ])
                    table.append(result_row)
                    return xr, table

            # Append regular row if not stopping yet
            table.append(row)

            # --- Update Interval (Step 3a/3b based on sign of f(xl)*f(xr)) ---
            test = fl * fr

            if test < 0: # Root is in the lower subinterval [xl, xr]
                xu = xr
                fu = fr # Update f(xu) with the value we just calculated
            elif test > 0: # Root is in the upper subinterval [xr, xu]
                xl = xr
                fl = fr # Update f(xl) with the value we just calculated
            # else: test == 0 handled by the exact root check above

        # --- Max Iterations Reached ---
        # This is executed if the loop finishes without meeting 'es' or finding an exact root
        result_row = OrderedDict([
            ("Message", f"Stopped by reaching maximum iterations: {imax}"),
            ("Status", "MAX_ITERATIONS"),
            ("Details", f"Consider increasing max_iter for more precision")
        ])
        table.append(result_row)
        
        # Add final highlighted result row - match bisection format exactly
        final_result_row = OrderedDict([
            ("Iteration", "Result"),
            ("Xl", ""),
            ("f(Xl)", ""),
            ("Xu", ""),
            ("f(Xu)", ""),
            ("Xr", self._round_value(xr, decimal_places)),
            ("f(Xr)", self._round_value(fr, decimal_places)),
            ("Error %", self._format_error(ea, decimal_places) if isinstance(ea, (int, float)) else "---")
        ])
        table.append(final_result_row)
        
        return xr, table # Return the last calculated approximation