from typing import Tuple, List, Dict, Optional, Any, Union
from src.core.methods import (BisectionMethod, FalsePositionMethod, 
                              FixedPointMethod, NewtonRaphsonMethod, SecantMethod,
                              GaussEliminationMethod, GaussEliminationPartialPivoting,
                              LUDecompositionMethod, LUDecompositionPartialPivotingMethod,
                              GaussJordanMethod, GaussJordanPartialPivotingMethod,
                              CramersRuleMethod)
from src.core.history import HistoryManager
import sympy as sp
import numpy as np
import logging
import re
import ast

class Solver:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.methods = {
            "Bisection": BisectionMethod(),
            "False Position": FalsePositionMethod(),
            "Fixed Point": FixedPointMethod(),
            "Newton-Raphson": NewtonRaphsonMethod(),
            "Secant": SecantMethod(),
            "Gauss Elimination": GaussEliminationMethod(),
            "Gauss Elimination (Partial Pivoting)": GaussEliminationPartialPivoting(),
            "LU Decomposition": LUDecompositionMethod(),
            "LU Decomposition (Partial Pivoting)": LUDecompositionPartialPivotingMethod(),
            "Gauss-Jordan": GaussJordanMethod(),
            "Gauss-Jordan (Partial Pivoting)": GaussJordanPartialPivotingMethod(),
            "Cramer's Rule": CramersRuleMethod()
        }
        
        # Method categories for guidance
        self.method_categories = {
            "root_finding": [
                "Bisection", 
                "False Position", 
                "Fixed Point", 
                "Newton-Raphson", 
                "Secant"
            ],
            "linear_system": [
                "Gauss Elimination", 
                "Gauss Elimination (Partial Pivoting)", 
                "LU Decomposition", 
                "LU Decomposition (Partial Pivoting)", 
                "Gauss-Jordan", 
                "Gauss-Jordan (Partial Pivoting)",
                "Cramer's Rule"
            ]
        }
        
        # Method recommendations based on problem characteristics
        self.method_recommendations = {
            "stable_linear_system": [
                "Gauss Elimination (Partial Pivoting)",
                "LU Decomposition (Partial Pivoting)",
                "Gauss-Jordan (Partial Pivoting)"
            ],
            "small_linear_system": [
                "Cramer's Rule"
            ],
            "well_conditioned_linear_system": [
                "Gauss Elimination",
                "LU Decomposition",
                "Gauss-Jordan"
            ],
            "smooth_function_with_derivative": [
                "Newton-Raphson"
            ],
            "function_with_bracket": [
                "Bisection",
                "False Position"
            ],
            "iterative_function": [
                "Fixed Point"
            ],
            "no_derivative_available": [
                "Secant"
            ]
        }
        
        self.MAX_EPS = 100.0  # Maximum allowed epsilon value
        self.settings = {
            "decimal_places": 4,
            "max_iterations": 100,
            "error_tolerance": 0.0001,
            "stop_by_eps": True
        }
        
        # Default settings
        self.decimal_places = 6
        self.max_iter = 50
        self.eps = 0.0001
        self.max_eps = 100.0  # Maximum allowed epsilon value
        self.stop_by_eps = True
        
        # Initialize history manager
        self.history_manager = HistoryManager()

    def validate_function(self, func: str) -> Optional[str]:
        """Validate the mathematical function expression."""
        try:
            # Clean up the function string
            func = func.strip()
            
            # Replace common math functions with sympy equivalents
            func = func.replace("math.sin", "sin")
            func = func.replace("math.cos", "cos")
            func = func.replace("math.tan", "tan")
            func = func.replace("math.log", "log")
            func = func.replace("math.log10", "log10")
            func = func.replace("math.exp", "exp")
            func = func.replace("math.sqrt", "sqrt")
            
            # Add multiplication operator between number and variable
            func = re.sub(r'(\d)x', r'\1*x', func)
            func = re.sub(r'x(\d)', r'x*\1', func)
            
            # Create symbolic variable and parse expression
            x = sp.symbols('x')
            expr = sp.sympify(func)
            
            # Test if the function can be evaluated
            expr.subs(x, 1.0)
            return None
        except Exception as e:
            self.logger.error(f"Function validation error: {str(e)}")
            if "could not parse" in str(e):
                return "Invalid mathematical expression. Please check the syntax."
            elif "invalid syntax" in str(e):
                return "Invalid syntax in the mathematical expression. Please check for missing operators or parentheses."
            else:
                return f"Error in function expression: {str(e)}"

    def validate_matrix_vector(self, matrix_str: str, vector_str: str) -> Optional[str]:
        """Validate the matrix and vector inputs for linear system methods with enhanced checks."""
        try:
            # Parse matrix and vector strings
            try:
                matrix = ast.literal_eval(matrix_str)
            except (SyntaxError, ValueError) as e:
                return f"Invalid matrix format: {str(e)}. Use proper Python list syntax like [[1, 2], [3, 4]]."
                
            try:
                vector = ast.literal_eval(vector_str)
            except (SyntaxError, ValueError) as e:
                return f"Invalid vector format: {str(e)}. Use proper Python list syntax like [5, 6]."
            
            # Check if the matrix is empty
            if not matrix:
                return "Matrix cannot be empty"
                
            # Check if the vector is empty
            if not vector:
                return "Vector cannot be empty"
            
            # Check if matrix is a list of lists
            if not all(isinstance(row, list) for row in matrix):
                return "Matrix must be a list of lists. Example: [[1, 2], [3, 4]]"
                
            # Check if vector is a list
            if not isinstance(vector, list):
                return "Vector must be a list. Example: [5, 6]"
            
            # Check if matrix rows have equal length
            row_lengths = [len(row) for row in matrix]
            if len(set(row_lengths)) > 1:
                return f"All rows in the matrix must have the same length. Current row lengths: {row_lengths}"
            
            # Convert to numpy arrays for validation
            try:
                A = np.array(matrix, dtype=np.float64)
                b = np.array(vector, dtype=np.float64)
            except ValueError as e:
                return f"Matrix or vector contains non-numeric values: {str(e)}"
            except Exception as e:
                return f"Error converting to numerical array: {str(e)}"
            
            # Check if matrix is square
            if A.shape[0] != A.shape[1]:
                return f"Matrix must be square. Current dimensions: {A.shape[0]}x{A.shape[1]}"
            
            # Check if dimensions match
            if A.shape[0] != len(b):
                return f"Matrix and vector dimensions do not match. Matrix rows: {A.shape[0]}, Vector length: {len(b)}"
            
            # Check for NaN or Inf values
            if np.any(np.isnan(A)):
                return "Matrix contains NaN (Not a Number) values"
                
            if np.any(np.isinf(A)):
                return "Matrix contains infinite values"
                
            if np.any(np.isnan(b)):
                return "Vector contains NaN (Not a Number) values"
                
            if np.any(np.isinf(b)):
                return "Vector contains infinite values"
            
            # Check for zeros on diagonal which can cause division by zero in some methods
            diagonal = np.diag(A)
            if np.any(np.abs(diagonal) < 1e-10):
                zeros_indices = np.where(np.abs(diagonal) < 1e-10)[0]
                return f"Matrix has zero or near-zero values on the diagonal at positions: {[i+1 for i in zeros_indices]}. This may cause division by zero in some methods."
            
            # Check if matrix is singular (determinant close to zero)
            try:
                # Use LU decomposition for better numerical stability in determinant calculation
                try:
                    from scipy import linalg
                    lu, piv = linalg.lu_factor(A)
                    det = linalg.det(lu) * np.prod(np.sign(piv - np.arange(len(piv))))
                except ImportError:
                    # Fallback to numpy if scipy not available
                    self.logger.warning("scipy not found. Using numpy for determinant calculation.")
                    det = np.linalg.det(A)
                
                if abs(det) < 1e-14:  # More strict threshold (was 1e-10)
                    try:
                        condition_number = np.linalg.cond(A)
                        if condition_number < 100:  # If condition number is reasonable, it might be solvable
                            self.logger.warning(f"Matrix has small determinant ({det:.2e}) but decent condition number ({condition_number:.2e}). It might be solvable.")
                            return None  # No error, allow to proceed with caution
                        return f"Matrix appears to be singular (determinant ≈ {det:.2e}, condition number ≈ {condition_number:.2e}). Some methods may not work or may produce inaccurate results."
                    except Exception:
                        return f"Matrix appears to be singular (determinant ≈ {det:.2e}). Some methods may not work or may produce inaccurate results."
            except Exception as e:
                # If we can't calculate determinant, check condition number instead
                try:
                    condition_number = np.linalg.cond(A)
                    if condition_number > 1e15:
                        return f"Matrix is extremely ill-conditioned (condition number ≈ {condition_number:.2e}). Results may be highly inaccurate."
                except Exception:
                    # If we can't calculate condition number either, provide a less specific warning
                    self.logger.warning(f"Could not verify matrix condition: {str(e)}")
                    # Continue without error - we'll try to solve anyway
            
            # Check for diagonal dominance which affects convergence of iterative methods
            diag_abs = np.abs(np.diag(A))
            row_sums = np.sum(np.abs(A), axis=1) - diag_abs
            is_diag_dominant = np.all(diag_abs >= row_sums)
            
            if not is_diag_dominant:
                # This is just a warning, not an error, so we don't return it
                self.logger.warning("Matrix is not diagonally dominant, which may affect convergence of some methods")
            
            # Check for symmetry which enables use of specialized methods
            is_symmetric = np.allclose(A, A.T, rtol=1e-10, atol=1e-10)
            if is_symmetric:
                self.logger.info("Matrix is symmetric, which enables use of specialized methods for symmetric matrices")
            
            return None
        except Exception as e:
            self.logger.error(f"Matrix/vector validation error: {str(e)}")
            if "could not parse" in str(e):
                return "Invalid matrix or vector format. Please use proper Python list syntax."
            elif "invalid syntax" in str(e):
                return "Invalid syntax in matrix or vector. Please check your input."
            elif "expected str" in str(e) or "cannot convert" in str(e):
                return "Matrix or vector contains non-numeric or incompatible values."
            else:
                return f"Error in matrix/vector input: {str(e)}"

    def validate_parameters(self, method_name: str, params: dict) -> Optional[str]:
        """Validate the parameters for the specific method."""
        try:
            if method_name in ["Bisection", "False Position"]:
                if not all(k in params for k in ["xl", "xu"]):
                    return "Missing parameters: xl and xu required"
                if not isinstance(params["xl"], (int, float)) or not isinstance(params["xu"], (int, float)):
                    return "xl and xu must be numbers"
                if params["xl"] >= params["xu"]:
                    return "xl must be less than xu"
            elif method_name in ["Fixed Point", "Newton-Raphson"]:
                if "xi" not in params:
                    return "Missing parameter: xi required"
                if not isinstance(params["xi"], (int, float)):
                    return "xi must be a number"
            elif method_name == "Secant":
                if not all(k in params for k in ["xi_minus_1", "xi"]):
                    return "Missing parameters: xi_minus_1 and xi required"
                if not isinstance(params["xi_minus_1"], (int, float)) or not isinstance(params["xi"], (int, float)):
                    return "xi_minus_1 and xi must be numbers"
                if params["xi_minus_1"] == params["xi"]:
                    return "xi_minus_1 must be different from xi"
            elif method_name == "Gauss Elimination":
                if not all(k in params for k in ["matrix", "vector"]):
                    return "Missing parameters: matrix and vector required"
                if not isinstance(params["matrix"], str) or not isinstance(params["vector"], str):
                    return "matrix and vector must be strings"
                matrix_error = self.validate_matrix_vector(params["matrix"], params["vector"])
                if matrix_error:
                    return matrix_error
            return None
        except Exception as e:
            self.logger.error(f"Parameter validation error: {str(e)}")
            return f"Parameter validation error: {str(e)}"

    def solve(self, method_name: str, func: str, params: dict, eps: float = None, eps_operator: str = "<=", max_iter: int = None, 
             stop_by_eps: bool = None, decimal_places: int = None) -> Tuple[Union[float, List[float], None], List[Dict]]:
        """
        Solve the equation using the specified method with enhanced error handling.
        
        Args:
            method_name: Name of the numerical method to use
            func: Mathematical function as a string
            params: Dictionary of parameters required by the method
            eps: Error tolerance (optional, uses default if not provided)
            eps_operator: Comparison operator for epsilon check ("<=", ">=", "<", ">", "=")
            max_iter: Maximum number of iterations (optional, uses default if not provided)
            stop_by_eps: Whether to stop by error tolerance (optional, uses default if not provided)
            decimal_places: Number of decimal places for rounding (optional, uses default if not provided)
            
        Returns:
            Tuple of (root, table_data) where root is the solution or None if not found,
            and table_data is a list of dictionaries containing iteration details
        """
        try:
            # Use default values if not provided
            eps = eps if eps is not None else self.eps
            max_iter = max_iter if max_iter is not None else self.max_iter
            stop_by_eps = stop_by_eps if stop_by_eps is not None else self.stop_by_eps
            decimal_places = decimal_places if decimal_places is not None else self.decimal_places

            # Validate inputs
            if method_name not in self.methods:
                return None, [{"Error": f"Unknown method: {method_name}"}]
                
            # Special handling for linear system methods
            if method_name in ["Gauss Elimination", "Gauss Elimination (Partial Pivoting)", 
                              "LU Decomposition", "LU Decomposition (Partial Pivoting)",
                              "Gauss-Jordan", "Gauss-Jordan (Partial Pivoting)",
                              "Cramer's Rule"]:
                # Extract matrix and vector from params
                matrix = params.get("matrix")
                vector = params.get("vector")
                
                if not matrix or not vector:
                    return None, [{"Error": "Matrix and vector are required for linear system methods"}]
                    
                # Validate matrix and vector
                validation_error = self.validate_matrix_vector(matrix, vector)
                if validation_error:
                    return None, [{"Error": validation_error}]
                
                # Call the method
                result, table = self.methods[method_name].solve(matrix, vector, decimal_places)
                
                # Save to history with a placeholder function name
                if result is not None:
                    # For linear system methods, use "System of Linear Equations" as the function name
                    self.history_manager.save_solution(
                        "System of Linear Equations",
                        method_name,
                        result,
                        table
                    )
                
                return result, table
            else:
                # Validate function
                func_error = self.validate_function(func)
                if func_error:
                    return None, [{"Error": func_error}]
                
                # Validate parameters
                param_error = self.validate_parameters(method_name, params)
                if param_error:
                    return None, [{"Error": param_error}]
                
                # Call the method
                if method_name == "Bisection" or method_name == "False Position":
                    xl = float(params.get("xl", 0))
                    xu = float(params.get("xu", 0))
                    # Pass stop_by_eps directly to control whether to stop by epsilon or iterations
                    result, table = self.methods[method_name].solve(func, xl, xu, eps, eps_operator, max_iter, stop_by_eps, decimal_places)
                elif method_name == "Fixed Point" or method_name == "Newton-Raphson":
                    xi = float(params.get("xi", 0))
                    # Pass stop_by_eps directly to control whether to stop by epsilon or iterations
                    if method_name == "Fixed Point":
                        # Add auto_generate_g parameter for Fixed Point method
                        auto_generate_g = params.get("auto_generate_g", False)
                        result, table = self.methods[method_name].solve(
                            func, xi, eps, eps_operator, max_iter, stop_by_eps, decimal_places, 
                            auto_generate_g=auto_generate_g
                        )
                    else:
                        # Newton-Raphson returns a NewtonRaphsonResult object
                        result_obj = self.methods[method_name].solve(
                            func, xi, eps, eps_operator, max_iter, stop_by_eps, decimal_places
                        )
                        # Pass the result object directly to keep the iterations_table
                        result = result_obj
                        
                        # For backwards compatibility and history saving
                        if hasattr(result_obj, 'iterations_table'):
                            table = result_obj.iterations_table.to_dict('records')
                        else:
                            # Fallback for older method versions
                            result, table = result_obj
                elif method_name == "Secant":
                    xi_minus_1 = float(params.get("xi_minus_1", 0))
                    xi = float(params.get("xi", 0))
                    # Secant method returns a SecantResult object
                    result_obj = self.methods[method_name].solve(
                        func, xi_minus_1, xi, eps, eps_operator, max_iter, stop_by_eps, decimal_places
                    )
                    # Pass the result object directly to keep the iterations_table
                    result = result_obj
                    
                    # For backwards compatibility and history saving
                    if hasattr(result_obj, 'iterations_table'):
                        table = result_obj.iterations_table.to_dict('records')
                    else:
                        # Fallback for older method versions
                        result, table = result_obj
                else:
                    # Fallback for any other methods
                    result, table = self.methods[method_name].solve(func, params, eps, eps_operator, max_iter, stop_by_eps, decimal_places)
                
                # Save to history
                if result is not None:
                    # For structured result objects, extract the root for saving to history
                    history_result = result
                    if hasattr(result, 'root'):
                        history_result = result.root
                    
                    self.history_manager.save_solution(
                        func,
                        method_name,
                        history_result,
                        table
                    )
                
                return result, table
                
        except Exception as e:
            self.logger.error(f"Solver error: {str(e)}")
            return None, [{"Error": f"Solver error: {str(e)}"}]

    def get_recommended_methods(self, problem_type: str, matrix_size: int = None, condition_number: float = None) -> List[str]:
        """
        Get recommended methods based on problem characteristics.
        
        Args:
            problem_type: Type of problem ('root_finding' or 'linear_system')
            matrix_size: Size of the matrix for linear systems
            condition_number: Condition number of the matrix for linear systems
            
        Returns:
            List of recommended method names
        """
        recommendations = []
        
        if problem_type == "linear_system":
            if matrix_size is not None and matrix_size <= 4:
                recommendations.extend(self.method_recommendations["small_linear_system"])
            
            if condition_number is not None:
                if condition_number < 100:
                    # Well-conditioned matrix
                    recommendations.extend(self.method_recommendations["well_conditioned_linear_system"])
                else:
                    # Ill-conditioned matrix
                    recommendations.extend(self.method_recommendations["stable_linear_system"])
            else:
                # No condition number available, recommend stable methods
                recommendations.extend(self.method_recommendations["stable_linear_system"])
        
        elif problem_type == "root_finding":
            # Default to general recommendations for root finding
            recommendations.extend(self.method_recommendations["function_with_bracket"])
            recommendations.extend(self.method_recommendations["smooth_function_with_derivative"])
            
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for method in recommendations:
            if method not in seen:
                seen.add(method)
                unique_recommendations.append(method)
                
        return unique_recommendations