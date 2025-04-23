from .base import NumericalMethodBase
from typing import Tuple, List, Dict
import numpy as np
import ast

class GaussEliminationMethod(NumericalMethodBase):
    def solve(self, matrix_str: str, vector_str: str, decimal_places: int = 6) -> Tuple[List[float], List[Dict]]:
        """
        Solve a system of linear equations using Gauss Elimination without row swapping.
        Following the book's approach exactly with enhanced numerical stability.
        
        Args:
            matrix_str: String representation of the coefficient matrix
            vector_str: String representation of the constants vector
            decimal_places: Number of decimal places for rounding
            
        Returns:
            Tuple containing the solution vector and a list of dictionaries with step details
        """
        try:
            # Parse matrix and vector strings
            matrix = ast.literal_eval(matrix_str)
            vector = ast.literal_eval(vector_str)
            
            # Convert to numpy arrays with higher precision
            A = np.array(matrix, dtype=np.float64)
            b = np.array(vector, dtype=np.float64)
            n = len(b)
            table = []
            
            # Check if matrix dimensions are valid
            if A.shape[0] != A.shape[1]:
                return None, [{"Error": f"Matrix must be square. Current dimensions: {A.shape[0]}x{A.shape[1]}"}]
            
            if A.shape[0] != len(b):
                return None, [{"Error": f"Matrix dimensions ({A.shape[0]}x{A.shape[1]}) do not match vector length ({len(b)})"}]
            
            # Add initial augmented matrix
            row = {
                "Step": "Initial System",
                "Matrix": self._format_augmented_matrix(A, b, decimal_places),
                "Operation": "Original augmented matrix [A|b]"
            }
            table.append(row)
            
            # Forward elimination
            for i in range(n):
                # Check for zero pivot
                if abs(A[i, i]) < 1e-10:
                    return None, [{"Error": f"Zero pivot encountered at row {i+1}. Try using partial pivoting."}]
                
                # Eliminate below
                for j in range(i + 1, n):
                    if abs(A[j, i]) > 1e-10:  # Only eliminate non-zero elements
                        # Calculate multiplier exactly as in the book
                        multiplier = A[j, i] / A[i, i]
                        
                        # Show multiplier calculation
                        row = {
                            "Step": f"Multiplier",
                            "Matrix": "",
                            "Operation": f"m{j+1}{i+1} = a{j+1}{i+1}/a{i+1}{i+1} = {self._format_value(A[j, i], decimal_places)}/{self._format_value(A[i, i], decimal_places)} = {self._format_value(multiplier, decimal_places)}"
                        }
                        table.append(row)
                        
                        # Store original values for verification
                        old_A = A[j].copy()
                        old_b = b[j]
                        
                        # Perform elimination exactly as in the book
                        for k in range(i, n):
                            A[j, k] = A[j, k] - multiplier * A[i, k]
                            
                            # Ensure very small values are set to zero (numerical stability)
                            if abs(A[j, k]) < 1e-14:
                                A[j, k] = 0.0
                                
                        b[j] = b[j] - multiplier * b[i]
                        
                        # Show the result
                        row = {
                            "Step": f"Elimination",
                            "Matrix": self._format_augmented_matrix(A, b, decimal_places),
                            "Operation": f"R{j+1} = R{j+1} - ({self._format_value(multiplier, decimal_places)})R{i+1}"
                        }
                        table.append(row)
            
            # Show the upper triangular system
            row = {
                "Step": "Upper Triangular",
                "Matrix": self._format_augmented_matrix(A, b, decimal_places),
                "Operation": "System is now in upper triangular form"
            }
            table.append(row)
            
            # Check for singular matrix (zero on diagonal)
            for i in range(n):
                if abs(A[i, i]) < 1e-10:
                    return None, table + [{"Error": "Matrix is singular. No unique solution exists."}]
            
            # Show equations before back substitution
            equations = []
            for i in range(n):
                eq = []
                for j in range(n):
                    if abs(A[i, j]) > 1e-10:  # Only include non-zero coefficients
                        if abs(A[i, j] - 1.0) < 1e-10:
                            eq.append(f"x{j+1}")
                        elif abs(A[i, j] + 1.0) < 1e-10:
                            eq.append(f"-x{j+1}")
                        else:
                            eq.append(f"{self._format_value(A[i, j], decimal_places)}x{j+1}")
                if not eq:
                    eq = ["0"]
                equations.append(" + ".join(eq).replace("+ -", "- ") + f" = {self._format_value(b[i], decimal_places)}")
            
            row = {
                "Step": "System of Equations",
                "Matrix": "\n".join(equations),
                "Operation": "Ready for back substitution"
            }
            table.append(row)
            
            # Back substitution exactly as in the book
            x = np.zeros(n, dtype=np.float64)
            back_sub_steps = []
            
            for i in range(n-1, -1, -1):
                # Calculate sum of known terms
                sum_term = 0
                for j in range(i+1, n):
                    sum_term += A[i, j] * x[j]
                
                # Calculate x[i]
                x[i] = (b[i] - sum_term) / A[i, i]
                
                # Format the back substitution step
                step_desc = f"x{i+1} = (b{i+1} - "
                sum_terms = []
                for j in range(i+1, n):
                    if abs(A[i, j]) > 1e-10:  # Only include non-zero coefficients
                        sum_terms.append(f"{self._format_value(A[i, j], decimal_places)}*{self._format_value(x[j], decimal_places)}")
                
                if sum_terms:
                    step_desc += " - ".join(sum_terms)
                else:
                    step_desc += "0"
                
                step_desc += f") / {self._format_value(A[i, i], decimal_places)} = {self._format_value(x[i], decimal_places)}"
                back_sub_steps.append(step_desc)
            
            # Add back substitution steps to the table
            row = {
                "Step": "Back Substitution",
                "Matrix": "\n".join(back_sub_steps),
                "Operation": "Solving for x values"
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
            
            for i in range(n):
                lhs = np.dot(original_A[i], x)
                rhs = original_b[i]
                residual = abs(lhs - rhs)
                verification.append(f"Equation {i+1}: LHS = {self._format_value(lhs, decimal_places)}, RHS = {self._format_value(rhs, decimal_places)}, Residual = {self._format_value(residual, decimal_places)}")
            
            row = {
                "Step": "Verification",
                "Matrix": "\n".join(verification),
                "Operation": "Substituting solution back into original equations"
            }
            table.append(row)
            
            return solution, table
            
        except Exception as e:
            self.logger.error(f"Error in Gauss Elimination: {str(e)}")
            return None, [{"Error": f"An error occurred: {str(e)}"}]

    def _format_augmented_matrix(self, A, b, decimal_places):
        """Format the augmented matrix [A|b] as a string for display."""
        n = len(b)
        result = []
        
        for i in range(n):
            row = []
            # Format A part
            for j in range(n):
                row.append(f"{self._format_value(A[i, j], decimal_places):>8}")
            # Add separator
            row.append("|") 
            # Format b part
            row.append(f"{self._format_value(b[i], decimal_places):>8}")
            result.append(" ".join(row))
        
        return "\n".join(result)

    def _format_value(self, value: float, decimal_places: int) -> str:
        """Format a numeric value, removing trailing zeros."""
        if abs(value) < 1e-10:
            return "0"
        elif abs(value - round(value)) < 1e-10:  # Check if effectively an integer
            return str(int(round(value)))
        else:
            str_value = f"{value:.{decimal_places}f}"
            return str_value.rstrip('0').rstrip('.') 

    def _round_value(self, value: float, decimal_places: int) -> float:
        """Round a numeric value to the specified number of decimal places."""
        return round(value, decimal_places)