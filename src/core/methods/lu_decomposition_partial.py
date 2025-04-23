from .base import NumericalMethodBase
from typing import Tuple, List, Dict
import numpy as np
import ast

class LUDecompositionPartialPivotingMethod(NumericalMethodBase):
    def solve(self, matrix_str: str, vector_str: str, decimal_places: int = 6) -> Tuple[List[float], List[Dict]]:
        """
        Solve a system of linear equations using LU Decomposition with Partial Pivoting.
        This method decomposes the coefficient matrix into lower and upper triangular matrices,
        with row pivoting for numerical stability.
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
            
            # Initialize permutation matrix (represented as a list of indices)
            P = list(range(n))
            
            # Initialize L and U matrices
            L = np.identity(n, dtype=np.float64)  # Lower triangular with 1's on diagonal
            U = np.copy(A)  # Will become upper triangular
            
            # Track row swaps for detailed steps
            row_swaps = []
            
            # LU Decomposition with Partial Pivoting
            for i in range(n):
                # Find pivot (maximum element in current column)
                pivot_row = i
                pivot_value = abs(U[i, i])
                
                for j in range(i + 1, n):
                    if abs(U[j, i]) > pivot_value:
                        pivot_row = j
                        pivot_value = abs(U[j, i])
                
                # If pivot is too small, matrix may be singular
                if pivot_value < 1e-10:
                    return None, [{"Error": "Matrix is singular or nearly singular."}]
                
                # Swap rows if necessary
                if pivot_row != i:
                    # Record the swap
                    row_swaps.append((i, pivot_row))
                    
                    # Swap rows in U
                    U[[i, pivot_row]] = U[[pivot_row, i]]
                    
                    # Swap rows in L up to column i-1
                    if i > 0:
                        L[[i, pivot_row], :i] = L[[pivot_row, i], :i]
                    
                    # Swap permutation indices
                    P[i], P[pivot_row] = P[pivot_row], P[i]
                    
                    # Show the row swap
                    row = {
                        "Step": f"Row Swap",
                        "Matrix": self._format_augmented_matrix(U, b[P], decimal_places),
                        "Operation": f"Swap R{i+1} with R{pivot_row+1} because |{self._format_value(U[i, i], decimal_places)}| > |{self._format_value(A[i, i], decimal_places)}|"
                    }
                    table.append(row)
                
                # Perform elimination for this column
                for j in range(i + 1, n):
                    # Calculate multiplier
                    multiplier = U[j, i] / U[i, i]
                    L[j, i] = multiplier
                    
                    # Show the multiplier calculation
                    row = {
                        "Step": f"Calculate m{j+1}{i+1}",
                        "Matrix": f"m{j+1}{i+1} = a{j+1}{i+1} / a{i+1}{i+1} = {self._format_value(U[j, i], decimal_places)} / {self._format_value(U[i, i], decimal_places)} = {self._format_value(multiplier, decimal_places)}",
                        "Operation": f"Multiplier for row {j+1}"
                    }
                    table.append(row)
                    
                    # Eliminate entries below pivot
                    U[j, i] = 0  # Set the element directly to zero to avoid floating-point errors
                    for k in range(i + 1, n):
                        U[j, k] -= multiplier * U[i, k]
                    
                    # Show the elimination step
                    row = {
                        "Step": f"Elimination",
                        "Matrix": self._format_augmented_matrix(U, b[P], decimal_places),
                        "Operation": f"R{j+1} = R{j+1} - ({self._format_value(multiplier, decimal_places)}) * R{i+1}"
                    }
                    table.append(row)
            
            # Show the L matrix
            row = {
                "Step": "L Matrix",
                "Matrix": self._format_matrix(L, decimal_places),
                "Operation": "Lower triangular matrix with multipliers"
            }
            table.append(row)
            
            # Show the U matrix
            row = {
                "Step": "U Matrix",
                "Matrix": self._format_matrix(U, decimal_places),
                "Operation": "Upper triangular matrix"
            }
            table.append(row)
            
            # Permute the right-hand side vector
            b_permuted = b[P]
            
            # Show the permuted right-hand side
            row = {
                "Step": "Permuted b",
                "Matrix": self._format_vector(b_permuted, decimal_places),
                "Operation": "Right-hand side vector after permutation"
            }
            table.append(row)
            
            # Forward substitution to solve Ly = b_permuted
            y = np.zeros(n, dtype=np.float64)
            for i in range(n):
                y[i] = b_permuted[i]
                for j in range(i):
                    y[i] -= L[i, j] * y[j]
                
                # Show the forward substitution step
                row = {
                    "Step": f"Forward Substitution",
                    "Matrix": f"y{i+1} = {self._format_value(b_permuted[i], decimal_places)}",
                    "Operation": f"Solving for y{i+1}"
                }
                if i > 0:
                    terms = []
                    for j in range(i):
                        if abs(L[i, j]) > 1e-10:
                            terms.append(f"{self._format_value(L[i, j], decimal_places)} * {self._format_value(y[j], decimal_places)}")
                    if terms:
                        row["Matrix"] += f" - ({' + '.join(terms)})"
                row["Matrix"] += f" = {self._format_value(y[i], decimal_places)}"
                table.append(row)
            
            # Show the intermediate vector y
            row = {
                "Step": "Forward Substitution Result",
                "Matrix": self._format_vector(y, decimal_places),
                "Operation": "Intermediate vector y from Ly = b"
            }
            table.append(row)
            
            # Back substitution to solve Ux = y
            x = np.zeros(n, dtype=np.float64)
            for i in range(n - 1, -1, -1):
                x[i] = y[i]
                for j in range(i + 1, n):
                    x[i] -= U[i, j] * x[j]
                x[i] /= U[i, i]
                
                # Show the back substitution step
                row = {
                    "Step": f"Back Substitution",
                    "Matrix": f"x{i+1} = ({self._format_value(y[i], decimal_places)}",
                    "Operation": f"Solving for x{i+1}"
                }
                terms = []
                for j in range(i + 1, n):
                    if abs(U[i, j]) > 1e-10:
                        terms.append(f"{self._format_value(U[i, j], decimal_places)} * {self._format_value(x[j], decimal_places)}")
                if terms:
                    row["Matrix"] += f" - ({' + '.join(terms)})"
                row["Matrix"] += f") / {self._format_value(U[i, i], decimal_places)} = {self._format_value(x[i], decimal_places)}"
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
            return None, [{"Error": f"Error in LU Decomposition with Partial Pivoting: {str(e)}"}]
    
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
