from .base import NumericalMethodBase
from typing import Tuple, List, Dict
import numpy as np
import ast

class GaussEliminationPartialPivoting(NumericalMethodBase):
    def solve(self, matrix_str: str, vector_str: str, decimal_places: int = 6) -> Tuple[List[float], List[Dict]]:
        """
        Solve a system of linear equations using Gauss Elimination with partial pivoting.
        This method improves numerical stability by selecting the largest pivot in each column.
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
            
            # Add initial augmented matrix
            row = {
                "Step": "Initial System",
                "Matrix": self._format_augmented_matrix(A, b, decimal_places),
                "Operation": "Original augmented matrix [A|b]"
            }
            table.append(row)
            
            # Forward elimination with partial pivoting
            for i in range(n):
                # Find pivot row (maximum absolute value in current column)
                pivot_row = i
                max_val = abs(A[i, i])
                for k in range(i + 1, n):
                    if abs(A[k, i]) > max_val:
                        max_val = abs(A[k, i])
                        pivot_row = k
                
                # Swap rows if necessary
                if pivot_row != i:
                    A[i], A[pivot_row] = A[pivot_row].copy(), A[i].copy()
                    b[i], b[pivot_row] = b[pivot_row], b[i]
                    
                    row = {
                        "Step": f"Row Swap",
                        "Matrix": self._format_augmented_matrix(A, b, decimal_places),
                        "Operation": f"Swap R{i+1} ↔ R{pivot_row+1} for better pivot"
                    }
                    table.append(row)
                
                # Check for zero pivot
                if abs(A[i, i]) < 1e-10:
                    return None, [{"Error": "Zero pivot encountered. System may be singular."}]
                
                # Show pivot selection
                row = {
                    "Step": f"Pivot",
                    "Matrix": "",
                    "Operation": f"Selected pivot: a{i+1}{i+1} = {self._format_value(A[i, i], decimal_places)}"
                }
                table.append(row)
                
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
            
            # Show equations before back substitution
            equations = []
            for i in range(n):
                eq = []
                for j in range(n):
                    if abs(A[i, j]) > 1e-10:
                        if A[i, j] == 1:
                            eq.append(f"x{j+1}")
                        elif A[i, j] == -1:
                            eq.append(f"-x{j+1}")
                        else:
                            eq.append(f"{self._format_value(A[i, j], decimal_places)}x{j+1}")
                equations.append(" + ".join(eq) + f" = {self._format_value(b[i], decimal_places)}")
            
            row = {
                "Step": "System of Equations",
                "Matrix": "\n".join(equations),
                "Operation": "Ready for back substitution"
            }
            table.append(row)
            
            # Back substitution exactly as in the book
            x = np.zeros(n, dtype=np.float64)
            for i in range(n-1, -1, -1):
                # Calculate sum of known terms
                sum_term = 0
                for j in range(i+1, n):
                    sum_term += A[i, j] * x[j]
                
                # Solve for x[i]
                x[i] = (b[i] - sum_term) / A[i, i]
                
                # Show the calculation
                if i < n-1:
                    terms = []
                    for j in range(i+1, n):
                        if abs(A[i, j]) > 1e-10:
                            terms.append(f"{self._format_value(A[i, j], decimal_places)}({self._format_value(x[j], decimal_places)})")
                    
                    if terms:
                        calc = f"x{i+1} = ({self._format_value(b[i], decimal_places)} - ({' + '.join(terms)})) / {self._format_value(A[i, i], decimal_places)}"
                    else:
                        calc = f"x{i+1} = {self._format_value(b[i], decimal_places)} / {self._format_value(A[i, i], decimal_places)}"
                else:
                    calc = f"x{i+1} = {self._format_value(b[i], decimal_places)} / {self._format_value(A[i, i], decimal_places)}"
                
                row = {
                    "Step": f"Back Substitution",
                    "Matrix": calc,
                    "Operation": f"x{i+1} = {self._format_value(x[i], decimal_places)}"
                }
                table.append(row)
            
            # Show final solution
            solution = [f"x{i+1} = {self._format_value(x[i], decimal_places)}" for i in range(n)]
            row = {
                "Step": "Solution",
                "Matrix": "\n".join(solution),
                "Operation": "Final solution"
            }
            table.append(row)
            
            return x.tolist(), table
            
        except Exception as e:
            return None, [{"Error": f"Error in Gauss Elimination with Partial Pivoting: {str(e)}"}]

    def _format_augmented_matrix(self, A: np.ndarray, b: np.ndarray, decimal_places: int) -> str:
        """Format the augmented matrix [A|b] for display."""
        n = A.shape[0]
        formatted = []
        for i in range(n):
            row = []
            # Format matrix part
            for j in range(n):
                value = self._format_value(A[i, j], decimal_places)
                row.append(f"{value:>15}")
            # Add separator
            row.append("|")
            # Format vector part
            value = self._format_value(b[i], decimal_places)
            row.append(f"{value:>15}")
            formatted.append("  ".join(row))
        return "\n\n".join(formatted)

    def _format_value(self, value: float, decimal_places: int) -> str:
        """Format a numeric value, removing trailing zeros."""
        if abs(value) < 1e-10:
            return "0"
        elif abs(value - round(value)) < 1e-10:  # Check if effectively an integer
            return str(int(round(value)))
        else:
            str_value = f"{value:.{decimal_places}f}"
            return str_value.rstrip('0').rstrip('.')