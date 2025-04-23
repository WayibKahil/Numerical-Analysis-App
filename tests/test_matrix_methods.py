import sys
import os

# Add the src directory to the Python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.solver import Solver

def test_matrix_method_names():
    """Test to verify that all matrix method names in the application are consistent."""
    print("Testing matrix method names consistency...")
    
    # Create a solver instance
    solver = Solver()
    
    # Get the matrix methods from the solver
    linear_system_methods = solver.method_categories.get("linear_system", [])
    
    # Expected method names
    expected_methods = [
        "Gauss Elimination", 
        "Gauss Elimination (Partial Pivoting)", 
        "LU Decomposition", 
        "LU Decomposition (Partial Pivoting)", 
        "Gauss-Jordan", 
        "Gauss-Jordan (Partial Pivoting)",
        "Cramer's Rule"
    ]
    
    # Check if all expected methods are in the solver
    missing_methods = [method for method in expected_methods if method not in linear_system_methods]
    if missing_methods:
        print(f"ERROR: Missing expected methods: {missing_methods}")
        return False
    
    # Check if there are any unexpected methods
    unexpected_methods = [method for method in linear_system_methods if method not in expected_methods]
    if unexpected_methods:
        print(f"ERROR: Unexpected methods found: {unexpected_methods}")
        return False
    
    # Check if all methods can be accessed in the solver's methods dictionary
    missing_implementations = [method for method in linear_system_methods if method not in solver.methods]
    if missing_implementations:
        print(f"ERROR: Methods without implementations: {missing_implementations}")
        return False
    
    print("All matrix method names are consistent!")
    return True

def test_gaussian_elimination():
    """Test the Gauss Elimination method with a simple system."""
    print("\nTesting Gauss Elimination method...")
    
    # Create a solver instance
    solver = Solver()
    
    # Define a simple system
    matrix_str = "[[4, 1, 2], [2, 5, 1], [1, 2, 4]]"
    vector_str = "[4, 5, 6]"
    
    # Solve using Gauss Elimination
    result, table_data = solver.solve(
        method_name="Gauss Elimination",
        func="System of Linear Equations",
        params={"matrix": matrix_str, "vector": vector_str},
        decimal_places=4
    )
    
    print(f"Solution: {result}")
    print("Iterations:")
    for row in table_data:
        print(row)
    
    if not result:
        print("ERROR: No solution found")
        return False
    
    print("Gauss Elimination test completed!")
    return True

if __name__ == "__main__":
    print("Matrix Methods Test")
    print("==================")
    
    name_test_passed = test_matrix_method_names()
    gauss_test_passed = test_gaussian_elimination()
    
    if name_test_passed and gauss_test_passed:
        print("\nAll tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome tests FAILED!")
        sys.exit(1) 