from .base import NumericalMethodBase
from typing import Tuple, List, Dict
import numpy as np
import ast

class GaussJordanMethod(NumericalMethodBase):
    def solve(self, matrix_str: str, vector_str: str, decimal_places: int = 6) -> Tuple[List[float], List[Dict]]:
        """
        Solve a system of linear equations using the Gauss-Jordan method.
        This method transforms the augmented matrix to reduced row echelon form (identity matrix).
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
            
            # Create augmented matrix [A|b]
            augmented = np.column_stack((A, b))
            
            # Add initial matrix and vector
            row = {
                "Step": "Initial System",
                "Matrix": self._format_augmented_matrix(A, b, decimal_places),
                "Operation": "Original system [A]{X} = {B}"
            }
            table.append(row)
            
            # Forward elimination (Gauss elimination part)
            for i in range(n):
                # Find pivot (maximum element in current column)
                pivot_row = i
                for j in range(i + 1, n):
                    if abs(augmented[j, i]) > abs(augmented[pivot_row, i]):
                        pivot_row = j
                
                # Swap rows if necessary
                if pivot_row != i:
                    augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
                    
                    # Show the row swap
                    row = {
                        "Step": f"Row Swap",
                        "Matrix": self._format_augmented_matrix_from_augmented(augmented, n, decimal_places),
                        "Operation": f"Swap R{i+1} with R{pivot_row+1} for better pivot"
                    }
                    table.append(row)
                
                # Normalize the pivot row
                pivot = augmented[i, i]
                if abs(pivot) < 1e-10:
                    return None, [{"Error": "Zero pivot encountered. System may be singular."}]
                
                if abs(pivot - 1.0) > 1e-10:  # Only normalize if not already 1
                    factor = 1.0 / pivot
                    augmented[i] *= factor
                    
                    # Show the normalization
                    row = {
                        "Step": f"Normalize R{i+1}",
                        "Matrix": self._format_augmented_matrix_from_augmented(augmented, n, decimal_places),
                        "Operation": f"R{i+1} = ({self._format_value(factor, decimal_places)}) * R{i+1}"
                    }
                    table.append(row)
                
                # Eliminate entries in column i
                for j in range(n):
                    if j != i:  # Skip the pivot row
                        factor = augmented[j, i]
                        if abs(factor) > 1e-10:  # Only eliminate if non-zero
                            augmented[j] -= factor * augmented[i]
                            
                            # Show the elimination
                            row = {
                                "Step": f"Eliminate x{i+1} from R{j+1}",
                                "Matrix": self._format_augmented_matrix_from_augmented(augmented, n, decimal_places),
                                "Operation": f"R{j+1} = R{j+1} - ({self._format_value(factor, decimal_places)}) * R{i+1}"
                            }
                            table.append(row)
            
            # Extract the solution
            x = augmented[:, n]
            
            # Show final solution
            solution = [f"x{i+1} = {self._format_value(x[i], decimal_places)}" for i in range(n)]
            row = {
                "Step": "Solution",
                "Matrix": "\n".join(solution),
                "Operation": "Final solution from the reduced row echelon form"
            }
            table.append(row)
            
            return x.tolist(), table
            
        except Exception as e:
            return None, [{"Error": f"Error in Gauss-Jordan method: {str(e)}"}]
    
    def _format_augmented_matrix_from_augmented(self, augmented: np.ndarray, n: int, decimal_places: int) -> str:
        """Format the augmented matrix for display."""
        rows = []
        for i in range(n):
            row = []
            # Format matrix part
            for j in range(n):
                value = self._format_value(augmented[i, j], decimal_places)
                row.append(f"{value:>15}")
            # Add separator
            row.append("|")
            # Format vector part
            value = self._format_value(augmented[i, n], decimal_places)
            row.append(f"{value:>15}")
            rows.append("  ".join(row))
        return "\n\n".join(rows)
    
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
