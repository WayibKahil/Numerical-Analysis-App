from .base import NumericalMethodBase
from typing import Tuple, List, Dict
import numpy as np
import ast

class LUDecompositionMethod(NumericalMethodBase):
    def solve(self, matrix_str: str, vector_str: str, decimal_places: int = 6) -> Tuple[List[float], List[Dict]]:
        """
        Solve a system of linear equations using LU Decomposition.
        This method decomposes the coefficient matrix into lower and upper triangular matrices,
        which is efficient for solving systems with the same coefficient matrix but different right-hand sides.
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
            
            # Add initial matrix and vector
            row = {
                "Step": "Initial System",
                "Matrix": self._format_augmented_matrix(A, b, decimal_places),
                "Operation": "Original system [A]{X} = {B}"
            }
            table.append(row)
            
            # Initialize L and U matrices
            L = np.identity(n, dtype=np.float64)  # Lower triangular with 1's on diagonal
            U = np.zeros((n, n), dtype=np.float64)  # Upper triangular
            
            # LU Decomposition
            for i in range(n):
                # Upper triangular elements (U)
                for j in range(i, n):
                    # Calculate U[i,j]
                    U[i, j] = A[i, j]
                    for k in range(i):
                        U[i, j] -= L[i, k] * U[k, j]
                
                # Lower triangular elements (L)
                for j in range(i + 1, n):
                    # Check for zero pivot
                    if abs(U[i, i]) < 1e-10:
                        return None, [{"Error": "Zero pivot encountered. System may be singular."}]
                    
                    # Calculate L[j,i]
                    L[j, i] = A[j, i]
                    for k in range(i):
                        L[j, i] -= L[j, k] * U[k, i]
                    L[j, i] /= U[i, i]
            
            # Show the L matrix
            row = {
                "Step": "L Matrix",
                "Matrix": self._format_matrix(L, decimal_places),
                "Operation": "Lower triangular matrix with 1's on diagonal"
            }
            table.append(row)
            
            # Show the U matrix
            row = {
                "Step": "U Matrix",
                "Matrix": self._format_matrix(U, decimal_places),
                "Operation": "Upper triangular matrix"
            }
            table.append(row)
            
            # Verify A = LU
            LU = np.matmul(L, U)
            row = {
                "Step": "Verification",
                "Matrix": self._format_matrix(LU, decimal_places),
                "Operation": "A = LU (should match the original matrix A)"
            }
            table.append(row)
            
            # Forward substitution to solve Ly = b
            y = np.zeros(n, dtype=np.float64)
            for i in range(n):
                y[i] = b[i]
                for j in range(i):
                    y[i] -= L[i, j] * y[j]
            
            # Show the intermediate vector y
            row = {
                "Step": "Forward Substitution",
                "Matrix": self._format_vector(y, decimal_places),
                "Operation": "Solving Ly = b for intermediate vector y"
            }
            table.append(row)
            
            # Back substitution to solve Ux = y
            x = np.zeros(n, dtype=np.float64)
            for i in range(n - 1, -1, -1):
                if abs(U[i, i]) < 1e-10:
                    return None, [{"Error": "Zero pivot encountered in back substitution."}]
                
                x[i] = y[i]
                for j in range(i + 1, n):
                    x[i] -= U[i, j] * x[j]
                x[i] /= U[i, i]
                
                # Show the calculation
                if i < n - 1:
                    terms = []
                    for j in range(i + 1, n):
                        if abs(U[i, j]) > 1e-10:
                            terms.append(f"{self._format_value(U[i, j], decimal_places)}({self._format_value(x[j], decimal_places)})")
                    
                    if terms:
                        calc = f"x{i+1} = ({self._format_value(y[i], decimal_places)} - ({' + '.join(terms)})) / {self._format_value(U[i, i], decimal_places)}"
                    else:
                        calc = f"x{i+1} = {self._format_value(y[i], decimal_places)} / {self._format_value(U[i, i], decimal_places)}"
                else:
                    calc = f"x{i+1} = {self._format_value(y[i], decimal_places)} / {self._format_value(U[i, i], decimal_places)}"
                
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
            return None, [{"Error": f"Error in LU Decomposition: {str(e)}"}]

    def _format_matrix(self, A: np.ndarray, decimal_places: int) -> str:
        """Format a matrix for display."""
        n = A.shape[0]
        formatted = []
        for i in range(n):
            row = []
            for j in range(n):
                value = self._format_value(A[i, j], decimal_places)
                row.append(f"{value:>15}")
            formatted.append("  ".join(row))
        return "\n\n".join(formatted)
    
    def _format_vector(self, v: np.ndarray, decimal_places: int) -> str:
        """Format a vector for display."""
        n = len(v)
        formatted = []
        for i in range(n):
            value = self._format_value(v[i], decimal_places)
            formatted.append(f"{value:>15}")
        return "\n\n".join(formatted)

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
