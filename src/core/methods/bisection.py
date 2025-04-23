from .base import NumericalMethodBase
from typing import Tuple, List, Dict, Callable, Optional
import numpy as np
from collections import OrderedDict

class BisectionMethod(NumericalMethodBase):
    """
    Implements the Bisection method based on the optimized algorithm 
    described in numerical methods literature (similar to Fig 1.4).
    
    This method finds a root of f(x) = 0 within a given interval [xl, xu],
    provided f(xl) and f(xu) have opposite signs. It minimizes function
    evaluations compared to simpler versions.
    """
    
    def solve(self, func_str: str, xl: float, xu: float, eps: float, eps_operator: str, max_iter: int, stop_by_eps: bool, decimal_places: int = 6,
              stop_criteria: str = "absolute", consecutive_check: bool = False, consecutive_tolerance: int = 3) -> Tuple[Optional[float], List[Dict]]:
        """
        Solve for the root using the optimized Bisection method (Fig 1.4).

        Args:
            func_str: The function f(x) as a string (e.g., "x**3 - x - 2").
            xl: Lower bound of the initial interval.
            xu: Upper bound of the initial interval.
            eps: Error tolerance (used if stop_by_eps is True)
            eps_operator: Comparison operator for epsilon check ("<=", ">=", "<", ">", "=")
            max_iter: Maximum number of iterations allowed.
            stop_by_eps: Whether to stop when error satisfies epsilon condition
            decimal_places: Number of decimal places for rounding in the output table.
            stop_criteria: Stopping criteria type ("absolute", "relative", "function", "interval")
            consecutive_check: Whether to also check for convergence over consecutive iterations
            consecutive_tolerance: Number of consecutive iterations within tolerance to confirm convergence

        Returns:
            Tuple containing:
            - The estimated root (float) if found within max_iter iterations and meeting criteria, 
              otherwise the last calculated approximation. Returns None if initial checks fail.
            - A list of dictionaries with iteration details.
        """
        table = []
        # Initialize variables as per Fig 1.4 pseudocode structure
        iter_count = 0 # Equivalent to 'iter' in pseudocode, starting from 0 for Python loop
        xr = xl        # Initialize xr to avoid unbound error if loop doesn't run
        xr_old = xl    # Initialize xr_old for first error calculation
        ea = np.inf    # Approximate relative error (εa), initialized to infinity
        
        # Use the appropriate stopping criteria based on parameters
        es = eps  # Use eps as our stopping criterion threshold
        imax = max_iter  # Use max_iter as our maximum iterations

        try:
            f = self._create_function(func_str)
            # Calculate f(xl) ONCE before the loop (Optimization - Fig 1.4)
            fl = float(f(xl)) 
            # Calculate f(xu) once for the initial check
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

        # --- Initial Checks (Step 1 & Pre-computation) ---
        # 1. Check bracketing condition: f(xl) * f(xu) < 0
        if fl * fu > 0:
            error_row = OrderedDict([
                ("Iteration", 0),
                ("Xl", self._round_value(xl, decimal_places)),
                ("f(Xl)", self._round_value(fl, decimal_places)),
                ("Xu", self._round_value(xu, decimal_places)),
                ("f(Xu)", self._round_value(fu, decimal_places)),
                ("Error", "Initial interval does not bracket a root."),
                ("Status", "BRACKETING_ERROR"),
                ("Details", f"f(xl)={fl:.{decimal_places}f} and f(xu)={fu:.{decimal_places}f} must have opposite signs.")
            ])
            return None, [error_row]

        # Optional: Check if bounds are roots (within tolerance)
        root_tolerance = 1e-12 
        if abs(fl) < root_tolerance:
            result_row = OrderedDict([
                 ("Iteration", 0), ("Xl", self._round_value(xl, decimal_places)), ("f(Xl)", self._round_value(fl, decimal_places)),
                 ("Xu", self._round_value(xu, decimal_places)), ("f(Xu)", self._round_value(fu, decimal_places)),
                 ("Message", f"Initial lower bound {xl} is already a root (f(xl) ≈ 0)."), ("Status", "SUCCESS"),
                 ("Root", self._round_value(xl, decimal_places))
            ])
            return xl, [result_row]
        if abs(fu) < root_tolerance:
             result_row = OrderedDict([
                 ("Iteration", 0), ("Xl", self._round_value(xl, decimal_places)), ("f(Xl)", self._round_value(fl, decimal_places)),
                 ("Xu", self._round_value(xu, decimal_places)), ("f(Xu)", self._round_value(fu, decimal_places)),
                 ("Message", f"Initial upper bound {xu} is already a root (f(xu) ≈ 0)."), ("Status", "SUCCESS"),
                 ("Root", self._round_value(xu, decimal_places))
             ])
             return xu, [result_row]

        # --- Iteration Loop (Equivalent to DO ... UNTIL in Fig 1.4) ---
        # We use range(max_iter) and break/return, which is equivalent to the DO loop with exit conditions.
        for i in range(max_iter): # Python loop from 0 to max_iter-1
            iter_count = i  # Start iteration count from 0 instead of 1
            
            xr_old = xr  # Store previous root estimate (xrold = xr)

            # Step 2: Calculate midpoint (new root estimate)
            xr = (xl + xu) / 2 

            try:
                # Calculate function value at midpoint (fr = f(xr)) - ONLY ONCE per iteration
                fr = float(f(xr)) 
            except Exception as e: # Handle evaluation errors during iteration
                 error_row = OrderedDict([
                     ("Iteration", iter_count), ("Xl", self._round_value(xl, decimal_places)), ("f(Xl)", self._round_value(fl, decimal_places)),
                     ("Xu", self._round_value(xu, decimal_places)), ("f(Xu)", self._round_value(fu, decimal_places)), # Show fu at this stage
                     ("Xr", self._round_value(xr, decimal_places)),
                     ("Error", f"Function evaluation failed at Xr={xr:.{decimal_places}f}. Details: {e}"), ("Status", "EVALUATION_ERROR")
                 ])
                 table.append(error_row)
                 return None, table 

            # Calculate approximate relative error (ea) - AFTER the first iteration
            if iter_count > 0 and abs(xr) > 1e-12: # Avoid division by zero, calculate after iter 0
                 ea = abs((xr - xr_old) / xr) * 100
            elif iter_count == 0:
                 ea = "---" # No error calculated on the first iteration
            else: # Handle xr close to zero case for ea calculation
                 ea = abs(xr - xr_old) * 100 # Or consider it large/undefined ("---")

            # Calculate error for different criteria
            if iter_count > 0:
                abs_diff = abs(xr - xr_old)
                if stop_criteria == "absolute":
                    error_value = abs_diff
                elif stop_criteria == "relative":
                    if abs(xr) < 1e-10:
                        error_value = abs_diff  # Use absolute error for very small values
                    else:
                        error_value = abs_diff / abs(xr) * 100  # Percentage relative error
                elif stop_criteria == "function":
                    error_value = abs(fr)  # Function value at current point
                elif stop_criteria == "interval":
                    error_value = abs(xu - xl)  # Width of the current interval
                else:
                    error_value = abs_diff  # Default to absolute error
            else:
                error_value = float('inf')  # No error for first iteration

            # --- Populate Table Row - Matching example columns ---
            row = OrderedDict([
                ("Iteration", iter_count),
                ("Xl", self._round_value(xl, decimal_places)),
                ("f(Xl)", self._round_value(fl, decimal_places)), # This is the stored fl
                ("Xu", self._round_value(xu, decimal_places)),
                ("f(Xu)", self._round_value(fu, decimal_places)), # This is the stored fu
                ("Xr", self._round_value(xr, decimal_places)),
                ("f(Xr)", self._round_value(fr, decimal_places)),
                ("Error %", self._format_error(ea, decimal_places) if iter_count > 0 else "---"), # Format error for display
            ])

            # Check if we found the exact root
            if abs(fr) < 1e-10:
                # The error 'ea' might be zero or non-zero depending on previous step
                ea = 0.0 # Set error to 0 if exact root is found
                table[-1]["εa (%)"] = self._format_error(ea, decimal_places) # Update last row error
                table[-1]["Status"] = "EXACT_ROOT"
                table[-1]["Details"] = f"f(Xr) = 0 found at iteration {iter_count}."
                
                # Add highlighted result row
                final_result_row = OrderedDict([
                    ("Iteration", "Result"),
                    ("Xl", ""),
                    ("f(Xl)", ""),
                    ("Xu", ""),
                    ("f(Xu)", ""),
                    ("Xr", self._round_value(xr, decimal_places)),
                    ("f(Xr)", self._round_value(fr, decimal_places)),
                    ("Error %", "0%"),
                    ("Status", "EXACT_ROOT")
                ])
                table.append(final_result_row)
                
                # Add final message row
                result_row = OrderedDict([
                    ("Message", f"Exact root found."),
                    ("Status", "SUCCESS"),
                    ("Details", f"f(Xr) = 0 at Xr = {self._round_value(xr, decimal_places)}."),
                    ("Root", self._round_value(xr, decimal_places))
                ])
                table.append(result_row)
                return xr, table

            # Check convergence criteria - only if stop_by_eps is True
            if iter_count > 0 and stop_by_eps:
                # For percentage-based epsilon (when eps > 1), use relative error
                if eps > 1:
                    # Calculate relative error as percentage
                    rel_error = abs_diff / abs(xr) * 100 if abs(xr) > 1e-10 else abs_diff
                    
                    # Direct comparison for relative error
                    if eps_operator == "<=":
                        if rel_error <= eps:
                            result_row = OrderedDict([
                                ("Message", f"Stopped by Epsilon: Relative Error {rel_error:.6f}% <= {eps}%"), 
                                ("Status", "CONVERGED"), 
                                ("Details", f"Achieved desired accuracy of {eps}%")
                            ])
                            table.append(row)
                            
                            # Add highlighted result row
                            final_result_row = OrderedDict([
                                ("Iteration", "Result"),
                                ("Xl", ""),
                                ("f(Xl)", ""),
                                ("Xu", ""),
                                ("f(Xu)", ""),
                                ("Xr", self._round_value(xr, decimal_places)),
                                ("f(Xr)", self._round_value(fr, decimal_places)),
                                ("Error %", self._format_error(rel_error, decimal_places)),
                                ("Status", "CONVERGED")
                            ])
                            table.append(final_result_row)
                            
                            table.append(result_row)
                            return xr, table
                    elif eps_operator == ">=":
                        if rel_error >= eps:
                            result_row = OrderedDict([
                                ("Message", f"Stopped by Epsilon: Relative Error {rel_error:.6f}% >= {eps}%"), 
                                ("Status", "STOPPED"), 
                                ("Details", f"Error threshold {eps}% reached")
                            ])
                            table.append(row)
                            table.append(result_row)
                            return xr, table
                    elif eps_operator == "<":
                        if rel_error < eps:
                            result_row = OrderedDict([
                                ("Message", f"Stopped by Epsilon: Relative Error {rel_error:.6f}% < {eps}%"), 
                                ("Status", "CONVERGED"), 
                                ("Details", f"Achieved desired accuracy of {eps}%")
                            ])
                            table.append(row)
                            
                            # Add highlighted result row
                            final_result_row = OrderedDict([
                                ("Iteration", "Result"),
                                ("Xl", ""),
                                ("f(Xl)", ""),
                                ("Xu", ""),
                                ("f(Xu)", ""),
                                ("Xr", self._round_value(xr, decimal_places)),
                                ("f(Xr)", self._round_value(fr, decimal_places)),
                                ("Error %", self._format_error(rel_error, decimal_places)),
                                ("Status", "CONVERGED")
                            ])
                            table.append(final_result_row)
                            
                            table.append(result_row)
                            return xr, table
                    elif eps_operator == ">":
                        if rel_error > eps:
                            result_row = OrderedDict([
                                ("Message", f"Stopped by Epsilon: Relative Error {rel_error:.6f}% > {eps}%"), 
                                ("Status", "STOPPED"), 
                                ("Details", f"Error exceeds threshold {eps}%")
                            ])
                            table.append(row)
                            table.append(result_row)
                            return xr, table
                    elif eps_operator == "=":
                        if abs(rel_error - eps) < 1e-10:
                            result_row = OrderedDict([
                                ("Message", f"Stopped by Epsilon: Relative Error {rel_error:.6f}% = {eps}%"), 
                                ("Status", "EXACT"), 
                                ("Details", f"Error exactly matches threshold {eps}%")
                            ])
                            table.append(row)
                            
                            # Add highlighted result row
                            final_result_row = OrderedDict([
                                ("Iteration", "Result"),
                                ("Xl", ""),
                                ("f(Xl)", ""),
                                ("Xu", ""),
                                ("f(Xu)", ""),
                                ("Xr", self._round_value(xr, decimal_places)),
                                ("f(Xr)", self._round_value(fr, decimal_places)),
                                ("Error %", self._format_error(rel_error, decimal_places)),
                                ("Status", "EXACT")
                            ])
                            table.append(final_result_row)
                            
                            table.append(result_row)
                            return xr, table
                else:
                    # Direct comparison for absolute error
                    if eps_operator == "<=":
                        if abs_diff <= eps:
                            result_row = OrderedDict([
                                ("Message", f"Stopped by Epsilon: |x{i} - x{i-1}| <= {eps}"), 
                                ("Status", "CONVERGED"), 
                                ("Details", f"Achieved desired accuracy of {eps}")
                            ])
                            table.append(row)
                            
                            # Add highlighted result row
                            final_result_row = OrderedDict([
                                ("Iteration", "Result"),
                                ("Xl", ""),
                                ("f(Xl)", ""),
                                ("Xu", ""),
                                ("f(Xu)", ""),
                                ("Xr", self._round_value(xr, decimal_places)),
                                ("f(Xr)", self._round_value(fr, decimal_places)),
                                ("Error %", self._format_error(abs_diff, decimal_places)),
                                ("Status", "CONVERGED")
                            ])
                            table.append(final_result_row)
                            
                            table.append(result_row)
                            return xr, table
                    elif eps_operator == ">=":
                        if abs_diff >= eps:
                            result_row = OrderedDict([
                                ("Message", f"Stopped by Epsilon: |x{i} - x{i-1}| >= {eps}"), 
                                ("Status", "STOPPED"), 
                                ("Details", f"Error threshold {eps} reached")
                            ])
                            table.append(row)
                            table.append(result_row)
                            return xr, table
                    elif eps_operator == "<":
                        if abs_diff < eps:
                            result_row = OrderedDict([
                                ("Message", f"Stopped by Epsilon: |x{i} - x{i-1}| < {eps}"), 
                                ("Status", "CONVERGED"), 
                                ("Details", f"Achieved desired accuracy of {eps}")
                            ])
                            table.append(row)
                            table.append(result_row)
                            return xr, table
                    elif eps_operator == ">":
                        if abs_diff > eps:
                            result_row = OrderedDict([
                                ("Message", f"Stopped by Epsilon: |x{i} - x{i-1}| > {eps}"), 
                                ("Status", "STOPPED"), 
                                ("Details", f"Error exceeds threshold {eps}")
                            ])
                            table.append(row)
                            table.append(result_row)
                            return xr, table
                    elif eps_operator == "=":
                        if abs(abs_diff - eps) < 1e-10:
                            result_row = OrderedDict([
                                ("Message", f"Stopped by Epsilon: |x{i} - x{i-1}| = {eps}"), 
                                ("Status", "EXACT"), 
                                ("Details", f"Error exactly matches threshold {eps}")
                            ])
                            table.append(row)
                            table.append(result_row)
                            return xr, table

            # Append the regular row if not stopping yet
            table.append(row)

            # --- Update Interval (Step 3a/3b based on test = fl * fr) ---
            test = fl * fr

            if test < 0: # Root is in the lower subinterval [xl, xr]
                xu = xr
                fu = fr
            else: # test > 0, Root is in the upper subinterval [xr, xu]
                xl = xr
                fl = fr # Update fl with the calculated fr (Key optimization from Fig 1.4)
                
            # Check Termination Conditions (IF ea < es OR iter >= imax EXIT) ---
            # Condition 1: Approximate relative error meets stopping criterion 'es'
            stop_by_error = False
            if isinstance(ea, (int, float)) and ea < es:
                stop_by_error = True
                row["Status"] = "CONVERGED"
                row["Details"] = f"εa ({ea:.{decimal_places}f}%) < eps ({es}%)"
                table.append(row) # Append the final successful iteration
                
                # Add a highlighted result row
                final_result_row = OrderedDict([
                    ("Iteration", "Result"),
                    ("Xl", ""),
                    ("f(Xl)", ""),
                    ("Xu", ""),
                    ("f(Xu)", ""),
                    ("Xr", self._round_value(xr, decimal_places)),
                    ("f(Xr)", self._round_value(fr, decimal_places)),
                    ("Error %", self._format_error(ea, decimal_places)),
                    ("Status", "CONVERGED")
                ])
                table.append(final_result_row)
                
                # Add final message row
                result_row = OrderedDict([
                    ("Message", f"Converged successfully."),
                    ("Status", "CONVERGED"),
                    ("Details", f"Stopped because εa ({ea:.{decimal_places}f}%) < eps ({es}%) at iteration {iter_count}."),
                    ("Root", self._round_value(xr, decimal_places))
                ])
                table.append(result_row)
                return xr, table
                
        # --- Max Iterations Reached ---
        result_row = OrderedDict([
            ("Message", f"Stopped by reaching maximum iterations: {max_iter}"),
            ("Status", "MAX_ITERATIONS"),
            ("Details", f"Consider increasing max_iter for more precision")
        ])
        table.append(result_row)
        
        # Add a final highlighted result row
        final_result_row = OrderedDict([
            ("Iteration", "Result"),
            ("Xl", ""),
            ("f(Xl)", ""),
            ("Xu", ""),
            ("f(Xu)", ""),
            ("Xr", self._round_value(xr, decimal_places)),
            ("f(Xr)", self._round_value(fr, decimal_places)),
            ("Error %", self._format_error(ea, decimal_places) if isinstance(ea, (int, float)) else "---"),
            ("Status", "FINAL_RESULT")
        ])
        table.append(final_result_row)
        
        return xr, table # Return the last approximation