from .base import NumericalMethodBase
from typing import Tuple, List, Dict
import numpy as np
import ast
import time
import math

class CramersRuleMethod(NumericalMethodBase):
    def _calculate_determinant_direct(self, matrix, size):
        """
        Calculate determinant directly for small matrices (2x2, 3x3).
        This is more educational and shows the step-by-step process.
        
        Args:
            matrix: The matrix to calculate the determinant for
            size: Size of the matrix (2 or 3)
            
        Returns:
            The determinant value and calculation steps as a string
        """
        steps = ""
        
        if size == 2:
            # For 2x2 matrix
            det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
            steps = f"det = ({matrix[0, 0]} × {matrix[1, 1]}) - ({matrix[0, 1]} × {matrix[1, 0]}) = {det}"
            
        elif size == 3:
            # For 3x3 matrix using the first row cofactors (similar to the C++ code)
            r0 = matrix[0, 0] * ((matrix[1, 1] * matrix[2, 2]) - (matrix[1, 2] * matrix[2, 1]))
            r1 = matrix[0, 1] * ((matrix[1, 0] * matrix[2, 2]) - (matrix[1, 2] * matrix[2, 0]))
            r2 = matrix[0, 2] * ((matrix[1, 0] * matrix[2, 1]) - (matrix[1, 1] * matrix[2, 0]))
            
            det = r0 - r1 + r2
            
            steps = (
                f"r0 = {matrix[0, 0]} × (({matrix[1, 1]} × {matrix[2, 2]}) - ({matrix[1, 2]} × {matrix[2, 1]})) = {r0}\n"
                f"r1 = {matrix[0, 1]} × (({matrix[1, 0]} × {matrix[2, 2]}) - ({matrix[1, 2]} × {matrix[2, 0]})) = {r1}\n"
                f"r2 = {matrix[0, 2]} × (({matrix[1, 0]} × {matrix[2, 1]}) - ({matrix[1, 1]} × {matrix[2, 0]})) = {r2}\n"
                f"det = r0 - r1 + r2 = {r0} - {r1} + {r2} = {det}"
            )
        else:
            # For larger matrices, return None to indicate we should use the library method
            return None, "Matrix too large for direct calculation"
            
        return det, steps

    def solve(self, matrix_str: str, vector_str: str, decimal_places: int = 6) -> Tuple[List[float], List[Dict]]:
        """
        Solve a system of linear equations using Cramer's Rule.
        
        Args:
            matrix_str: String representation of the coefficient matrix
            vector_str: String representation of the constants vector
            decimal_places: Number of decimal places for rounding
            
        Returns:
            Tuple containing the solution vector and a list of dictionaries with step details
        """
        try:
            # Track performance
            self._start_timer()
            start_time = time.time()
            
            # Parse matrix and vector strings
            try:
                matrix = ast.literal_eval(matrix_str)
                vector = ast.literal_eval(vector_str)
            except (SyntaxError, ValueError) as e:
                self.logger.error(f"Error parsing matrix or vector: {str(e)}")
                return None, [{"Error": f"Invalid matrix or vector format: {str(e)}"}]
            
            # Convert to numpy arrays with higher precision
            try:
                A = np.array(matrix, dtype=np.float64)
                b = np.array(vector, dtype=np.float64)
                n = len(b)
            except Exception as e:
                self.logger.error(f"Error converting to numpy arrays: {str(e)}")
                return None, [{"Error": f"Error converting input data: {str(e)}"}]
                
            table = []
            
            # Verify limitations of Cramer's Rule
            if n > 10:
                row = {
                    "Step": "Error",
                    "Matrix": f"Matrix size {n}x{n} is too large",
                    "Operation": "Cramer's rule is not recommended for matrices larger than 10x10 due to computational complexity."
                }
                table.append(row)
                return None, table
                
            # Check if matrix dimensions are valid
            if A.shape[0] != A.shape[1]:
                return None, [{"Error": f"Matrix must be square. Current dimensions: {A.shape[0]}x{A.shape[1]}"}]
            
            if A.shape[0] != len(b):
                return None, [{"Error": f"Matrix dimensions ({A.shape[0]}x{A.shape[1]}) do not match vector length ({len(b)})"}]
            
            # Performance warning for large matrices
            if n > 4:
                row = {
                    "Step": "Performance Warning",
                    "Matrix": f"Matrix size is {n}x{n}",
                    "Operation": "Cramer's rule is computationally expensive for large matrices. Consider using another method for better performance."
                }
                table.append(row)
            
            # Add educational note about Cramer's Rule
            row = {
                "Step": "Method Information",
                "Matrix": "Cramer's Rule",
                "Operation": "Cramer's rule uses determinants to solve linear systems. For a system Ax=b, each variable x_i = det(A_i)/det(A), where A_i is A with column i replaced by b."
            }
            table.append(row)
            
            # Add initial system
            row = {
                "Step": "Initial System",
                "Matrix": self._format_augmented_matrix(A, b, decimal_places),
                "Operation": "Original system [A|b]"
            }
            table.append(row)
            
            # Calculate the determinant of A - use direct method for small matrices for educational purposes
            use_direct_method = n <= 3  # Only use direct method for 2x2 and 3x3 matrices
            
            if use_direct_method:
                det_A, det_steps = self._calculate_determinant_direct(A, n)
                
                row = {
                    "Step": "Determinant of A (Direct Calculation)",
                    "Matrix": det_steps,
                    "Operation": "Calculate determinant using cofactor expansion"
                }
                table.append(row)
            else:
                # Use library methods for larger matrices
                try:
                    # Use LU decomposition for better numerical stability in determinant calculation
                    try:
                        from scipy import linalg
                        lu, piv = linalg.lu_factor(A)
                        det_A = linalg.det(lu) * np.prod(np.sign(piv - np.arange(len(piv))))
                    except (ImportError, NameError):
                        # Fallback to numpy if scipy not available
                        self.logger.warning("scipy not found. Using numpy for determinant calculation.")
                        det_A = np.linalg.det(A)
                except Exception as e:
                    self.logger.error(f"Error in determinant calculation: {str(e)}")
                    det_A = np.linalg.det(A)  # Final fallback
            
            # Format for display in scientific notation if very small
            det_A_display = self._format_value(det_A, decimal_places)
            
            # Add determinant value
            if not use_direct_method:
                row = {
                    "Step": "Determinant of A",
                    "Matrix": det_A_display,
                    "Operation": "Calculate determinant of coefficient matrix"
                }
                table.append(row)
            
            # Check if the determinant is near zero (system has no unique solution)
            if abs(det_A) < 1e-14:  # Using a stricter threshold of 1e-14 instead of 1e-10
                condition_number = np.linalg.cond(A)
                
                # Check if it's a "borderline" case that might still be solvable
                if abs(det_A) > 0 and condition_number < 100:
                    self.logger.warning(f"Matrix has very small determinant ({det_A_display}) but might still be solvable. Proceeding with caution.")
                    
                    # Add warning to the table
                    row = {
                        "Step": "Warning",
                        "Matrix": f"Determinant ≈ {det_A_display}",
                        "Operation": "Matrix is nearly singular, but we'll attempt to solve anyway. Results may be inaccurate."
                    }
                    table.append(row)
                else:
                    return None, table + [{
                        "Error": f"Determinant of coefficient matrix is effectively zero ({det_A_display}). "
                                f"No unique solution exists. Condition number: {condition_number:.2e}"
                    }]
            
            # Add condition number information
            condition_number = np.linalg.cond(A)
            row = {
                "Step": "Condition Number",
                "Matrix": f"{condition_number:.2e}",
                "Operation": f"Matrix condition number (measure of numerical stability, lower is better)"
            }
            table.append(row)
            
            if condition_number > 1e6:
                row = {
                    "Step": "Stability Warning",
                    "Matrix": "",
                    "Operation": "Matrix is ill-conditioned. Results may have reduced accuracy."
                }
                table.append(row)
            
            # Add recommendation for ill-conditioned matrices
            if condition_number > 100:
                row = {
                    "Step": "Recommendation",
                    "Matrix": "",
                    "Operation": "For this ill-conditioned matrix, consider using Gauss Elimination with Partial Pivoting or LU Decomposition with Partial Pivoting for better numerical stability."
                }
                table.append(row)
            
            # Prepare solution vector and explanation for each variable
            try:
                x = np.zeros(n, dtype=np.float64)
                cramer_steps = []
                
                # For each variable, create the matrix with replaced column and calculate determinant
                for i in range(n):
                    # Create a copy of the original matrix
                    A_i = A.copy()
                    
                    # Replace column i with the constant vector (similar to C++ code)
                    try:
                        A_i[:, i] = b
                    except Exception as e:
                        self.logger.error(f"Error replacing column {i}: {str(e)}")
                        return None, table + [{"Error": f"Error replacing column {i}: {str(e)}"}]
                    
                    # Format the matrix for display
                    display_matrix = self._format_matrix(A_i, decimal_places)
                    row = {
                        "Step": f"A{i+1} Matrix",
                        "Matrix": display_matrix,
                        "Operation": f"Replace column {i+1} of A with b"
                    }
                    table.append(row)
                    
                    # Calculate determinant of modified matrix - use direct method for small matrices
                    if use_direct_method:
                        det_A_i, det_steps = self._calculate_determinant_direct(A_i, n)
                        
                        row = {
                            "Step": f"Det(A{i+1}) (Direct Calculation)",
                            "Matrix": det_steps,
                            "Operation": f"Calculate determinant of A{i+1} using cofactor expansion"
                        }
                        table.append(row)
                    else:
                        # Use library methods for larger matrices
                        try:
                            # Use LU decomposition for better numerical stability
                            try:
                                lu, piv = linalg.lu_factor(A_i)
                                det_A_i = linalg.det(lu) * np.prod(np.sign(piv - np.arange(len(piv))))
                            except (ImportError, NameError):
                                # Fallback to numpy if scipy/linalg not available/defined
                                self.logger.warning("scipy.linalg not available. Using numpy for determinant calculation.")
                                det_A_i = np.linalg.det(A_i)
                        except Exception as e:
                            self.logger.error(f"Error in determinant calculation for A{i+1}: {str(e)}")
                            det_A_i = np.linalg.det(A_i)  # Final fallback
                        
                        det_A_i_display = self._format_value(det_A_i, decimal_places)
                        
                        row = {
                            "Step": f"Det(A{i+1})",
                            "Matrix": det_A_i_display,
                            "Operation": f"Calculate determinant of A{i+1}"
                        }
                        table.append(row)
                    
                    # Calculate x_i using Cramer's rule: x_i = det(A_i) / det(A)
                    try:
                        # For near-zero determinants, use a regularized approach
                        if abs(det_A) < 1e-10:
                            # Apply Tikhonov regularization - add a small value to the diagonal
                            # This stabilizes the division for ill-conditioned matrices
                            regularization = 1e-10
                            self.logger.warning(f"Using regularization for stability (det_A is very small)")
                            x[i] = det_A_i / (det_A + regularization * np.sign(det_A))
                            
                            # Add note about regularization
                            if i == 0:  # Only add this note once
                                row = {
                                    "Step": "Regularization Applied",
                                    "Matrix": f"det(A) ≈ {det_A_display}",
                                    "Operation": "Using numerical stabilization for very small determinant to avoid division by near-zero."
                                }
                                table.append(row)
                        else:
                            x[i] = det_A_i / det_A
                    except Exception as e:
                        self.logger.error(f"Error calculating x{i+1}: {str(e)}")
                        return None, table + [{"Error": f"Error calculating x{i+1}: {str(e)}"}]
                    
                    # Format the calculation step
                    det_A_i_display = self._format_value(det_A_i, decimal_places)
                    step_desc = f"x{i+1} = Det(A{i+1}) / Det(A) = {det_A_i_display} / {det_A_display} = {self._format_value(x[i], decimal_places)}"
                    cramer_steps.append(step_desc)
            except Exception as e:
                self.logger.error(f"Error in Cramer's rule calculation: {str(e)}")
                return None, table + [{"Error": f"Error in Cramer's rule calculation: {str(e)}"}]
            
            # Add Cramer's rule calculation steps to the table
            row = {
                "Step": "Cramer's Rule Calculations",
                "Matrix": "\n".join(cramer_steps),
                "Operation": "Calculate x values using Cramer's rule: x_i = Det(A_i) / Det(A)"
            }
            table.append(row)
            
            # Add final solution
            solution = [float(self._round_value(val, decimal_places)) for val in x]
            row = {
                "Step": "Solution",
                "Matrix": ", ".join([f"x{i+1} = {self._format_value(val, decimal_places)}" for i, val in enumerate(solution)]),
                "Operation": "Final solution vector"
            }
            table.append(row)
            
            # Verify solution by substituting back into original equations
            verification = []
            original_A = np.array(matrix, dtype=np.float64)
            original_b = np.array(vector, dtype=np.float64)
            max_residual = 0
            
            for i in range(n):
                lhs = np.dot(original_A[i], x)
                rhs = original_b[i]
                residual = abs(lhs - rhs)
                max_residual = max(max_residual, residual)
                verification.append(f"Equation {i+1}: LHS = {self._format_value(lhs, decimal_places)}, RHS = {self._format_value(rhs, decimal_places)}, Residual = {self._format_value(residual, decimal_places)}")
            
            row = {
                "Step": "Verification",
                "Matrix": "\n".join(verification),
                "Operation": "Substituting solution back into original equations"
            }
            table.append(row)
            
            # Stop the timer but don't add performance metrics to table
            self._stop_timer()
            
            return solution, table
            
        except Exception as e:
            self.logger.error(f"Error in Cramer's Rule: {str(e)}")
            return None, [{"Error": f"An error occurred: {str(e)}"}]

    def _format_augmented_matrix(self, A, b, decimal_places):
        """Format the augmented matrix [A|b] as a string for display."""
        n = len(b)
        result = []
        
        # Determine the width needed for each column
        max_width = 0
        for i in range(n):
            for j in range(n):
                formatted = self._format_value(A[i, j], decimal_places)
                max_width = max(max_width, len(formatted))
            
            formatted_b = self._format_value(b[i], decimal_places)
            max_width = max(max_width, len(formatted_b))
        
        # Add padding for readability
        col_width = max_width + 2
        
        
        
        # Format each row
        for i in range(n):
            row = ["|"]
            # Format A part
            for j in range(n):
                formatted = self._format_value(A[i, j], decimal_places)
                row.append(f"{formatted:^{col_width}}")
            # Add separator
            row.append("|") 
            # Format b part
            formatted_b = self._format_value(b[i], decimal_places)
            row.append(f"{formatted_b:^{col_width}}")
            row.append("|")
            result.append("".join(row))
            
            # Add horizontal line after each row
            
        
        return "\n".join(result)
    
    def _format_matrix(self, A, decimal_places):
        """Format matrix as a string for display."""
        n = A.shape[0]
        result = []
        
        # Determine the width needed for each column
        max_width = 0
        for i in range(n):
            for j in range(n):
                formatted = self._format_value(A[i, j], decimal_places)
                max_width = max(max_width, len(formatted))
                
        # Add padding for readability
        col_width = max_width + 2
        
        # Create a horizontal line for better readability
       
        
        # Format each row
        for i in range(n):
            row = ["|"]
            for j in range(n):
                formatted = self._format_value(A[i, j], decimal_places)
                row.append(f"{formatted:^{col_width}}")
            row.append("|")
            result.append("".join(row))
            
            # Add horizontal line after each row
           
        
        return "\n".join(result)

    def _format_value(self, value: float, decimal_places: int) -> str:
        """Format a numeric value, removing trailing zeros and handling special cases."""
        if abs(value) < 1e-10:
            return "0"
        elif abs(value) < 1e-4 or abs(value) > 1e6:  # Use scientific notation for very small or large values
            return f"{value:.{decimal_places}e}"
        elif abs(value - round(value)) < 1e-10:  # Check if effectively an integer
            return str(int(round(value)))
        else:
            str_value = f"{value:.{decimal_places}f}"
            return str_value.rstrip('0').rstrip('.') 

    def _round_value(self, value: float, decimal_places: int) -> float:
        """Round a numeric value to the specified number of decimal places with improved precision."""
        try:
            # Handle non-numeric or special values
            if value is None:
                return 0.0
                
            if not isinstance(value, (int, float)):
                return float(value) if value is not None else 0.0
                
            # Handle NaN and infinity
            if math.isnan(value):
                return 0.0
                
            if math.isinf(value):
                return float('inf') if value > 0 else float('-inf')
                
            # For very small values near zero, just return zero
            if abs(value) < 1e-14:
                return 0.0
                
            # For values very close to integers, snap to the integer
            if abs(value - round(value)) < 1e-10:
                return float(round(value))
                
            # Standard rounding with specified decimal places
            rounded = round(value, decimal_places)
            return rounded
            
        except Exception as e:
            self.logger.error(f"Error in _round_value: {str(e)}")
            # Safe fallback
            try:
                return float(round(float(value), decimal_places))
            except:
                return 0.0 