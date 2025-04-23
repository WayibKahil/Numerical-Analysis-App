import unittest
import sys
import os
import math
import numpy as np

# Add the src directory to the path so we can import modules from the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the numerical methods from the core package
from src.core.methods import (
    BisectionMethod,
    FalsePositionMethod,
    FixedPointMethod,
    NewtonRaphsonMethod,
    SecantMethod,
    GaussEliminationMethod
)

# Try to import optional methods - these might not be available yet
# so we handle the ImportError gracefully
try:
    from src.core.methods import LUDecompositionMethod
    has_lu = True
except ImportError:
    has_lu = False
    print("Note: LU Decomposition method not available for testing")

try:
    from src.core.methods import GaussJordanMethod
    has_gauss_jordan = True
except ImportError:
    has_gauss_jordan = False
    print("Note: Gauss-Jordan method not available for testing")


class TestRootFindingMethods(unittest.TestCase):
    """
    Test cases for root-finding numerical methods.
    
    This test class verifies the functionality of all root-finding methods:
    - Bisection Method
    - False Position Method
    - Fixed Point Method
    - Newton-Raphson Method
    - Secant Method
    
    Each test focuses on a specific method with simple test cases that have known solutions.
    """
    def setUp(self):
        """Set up test fixtures before each test method is run."""
        # Initialize all method instances
        self.bisection = BisectionMethod()
        self.false_position = FalsePositionMethod()
        self.fixed_point = FixedPointMethod()
        self.newton_raphson = NewtonRaphsonMethod()
        self.secant = SecantMethod()
        
        # Common parameters for all tests
        self.eps = 0.0001  # Error tolerance
        self.eps_operator = "<="  # Comparison operator for epsilon
        self.max_iter = 50  # Maximum iterations
        self.stop_by_eps = True  # Stop when error satisfies epsilon condition
        self.decimal_places = 6  # Decimal places for rounding

    def test_bisection_method(self):
        """
        Test the Bisection method with a quadratic function (x^2 - 4 = 0).
        Expected root: x = 2.0
        """
        # Test case: Simple quadratic function with root at x = 2
        root, table = self.bisection.solve(
            "x**2 - 4",  # Function: x^2 - 4 = 0
            0, 3,  # Interval [0, 3] contains the root x = 2
            self.eps, self.eps_operator, self.max_iter, self.stop_by_eps, self.decimal_places
        )
        # Verify the result is close to the expected root (2.0)
        self.assertAlmostEqual(root, 2.0, places=4)
        # Verify that the method generated iteration steps
        self.assertTrue(len(table) > 0)

    def test_bisection_cubic_function(self):
        """
        Test the Bisection method with a cubic function (x^3 - x - 2 = 0).
        Expected root: x ≈ 1.521
        """
        # Test case: Cubic function with root at x ≈ 1.521
        root, table = self.bisection.solve(
            "x**3 - x - 2",  # Function: x^3 - x - 2 = 0
            1, 2,  # Interval [1, 2] contains the root
            self.eps, self.eps_operator, self.max_iter, self.stop_by_eps, self.decimal_places
        )
        # Verify the result is close to the expected root (≈1.521)
        self.assertAlmostEqual(root, 1.521, places=3)
        # Verify that the method generated iteration steps
        self.assertTrue(len(table) > 0)

    def test_bisection_invalid_interval(self):
        """
        Test the Bisection method with an invalid interval that doesn't contain a root.
        Expected result: None with an error message.
        """
        # Test case: Invalid interval (no root in [3, 4] for x^2 - 4 = 0)
        root, table = self.bisection.solve(
            "x**2 - 4",  # Function: x^2 - 4 = 0
            3, 4,  # Interval [3, 4] doesn't contain any root
            self.eps, self.eps_operator, self.max_iter, self.stop_by_eps, self.decimal_places
        )
        # Verify that the method returns None for invalid intervals
        self.assertIsNone(root)
        # Verify that an error message is included in the results
        self.assertTrue(any("Error" in str(item) for item in table))

    def test_false_position_method(self):
        """
        Test the False Position method with a quadratic function (x^2 - 4 = 0).
        Expected root: x = 2.0
        """
        # Test case: Simple quadratic function with root at x = 2
        root, table = self.false_position.solve(
            "x**2 - 4",  # Function: x^2 - 4 = 0
            0, 3,  # Interval [0, 3] contains the root x = 2
            self.eps, self.eps_operator, self.max_iter, self.stop_by_eps, self.decimal_places
        )
        # Verify the result is close to the expected root (2.0)
        self.assertAlmostEqual(root, 2.0, places=4)
        # Verify that the method generated iteration steps
        self.assertTrue(len(table) > 0)

    def test_false_position_exponential(self):
        """
        Test the False Position method with an exponential function (e^x - 3 = 0).
        Expected root: x = ln(3) ≈ 1.099
        """
        # Test case: Exponential function with root at x = ln(3) ≈ 1.099
        root, table = self.false_position.solve(
            "exp(x) - 3",  # Function: e^x - 3 = 0
            0, 2,  # Interval [0, 2] contains the root
            self.eps, self.eps_operator, self.max_iter, self.stop_by_eps, self.decimal_places
        )
        # Verify the result is close to the expected root (ln(3) ≈ 1.099)
        self.assertAlmostEqual(root, math.log(3), places=4)
        # Verify that the method generated iteration steps
        self.assertTrue(len(table) > 0)

    def test_fixed_point_method(self):
        """
        Test the Fixed Point method with a function that has a fixed point at x = 2.
        Function: g(x) = 0.5*x + 1, which has a fixed point at x = 2.
        """
        # Test case: Function with fixed point at x = 2
        root, table = self.fixed_point.solve(
            "0.5*x + 1",  # Function: g(x) = 0.5*x + 1
            1,  # Initial guess
            self.eps, self.eps_operator, self.max_iter, self.stop_by_eps, self.decimal_places
        )
        # Verify the result is close to the expected fixed point (2.0)
        self.assertAlmostEqual(root, 2.0, places=4)
        # Verify that the method generated iteration steps
        self.assertTrue(len(table) > 0)

    def test_fixed_point_sqrt(self):
        """
        Test the Fixed Point method with a square root function.
        Function: g(x) = sqrt(10-x), which has a fixed point at approximately 2.7016.
        """
        # Test case: Function with fixed point at approximately 2.7016
        root, table = self.fixed_point.solve(
            "sqrt(10-x)",  # Function: g(x) = sqrt(10-x)
            3,  # Initial guess
            self.eps, self.eps_operator, self.max_iter, self.stop_by_eps, self.decimal_places
        )
        # Verify the result is close to the expected fixed point (≈2.7016)
        self.assertAlmostEqual(root, 2.7016, places=4)
        # Verify that the method generated iteration steps
        self.assertTrue(len(table) > 0)

    def test_newton_raphson_method(self):
        """
        Test the Newton-Raphson method with a quadratic function (x^2 - 4 = 0).
        Expected root: x = 2.0
        """
        # Test case: Simple quadratic function with root at x = 2
        root, table = self.newton_raphson.solve(
            "x**2 - 4",  # Function: x^2 - 4 = 0
            3,  # Initial guess
            self.eps, self.eps_operator, self.max_iter, self.stop_by_eps, self.decimal_places
        )
        # Verify the result is close to the expected root (2.0)
        self.assertAlmostEqual(root, 2.0, places=4)
        # Verify that the method generated iteration steps
        self.assertTrue(len(table) > 0)

    def test_newton_raphson_exponential(self):
        """
        Test the Newton-Raphson method with an exponential function (e^x - 2 = 0).
        Expected root: x = ln(2) ≈ 0.693
        """
        # Test case: Exponential function with root at x = ln(2) ≈ 0.693
        root, table = self.newton_raphson.solve(
            "exp(x) - 2",  # Function: e^x - 2 = 0
            0,  # Initial guess
            self.eps, self.eps_operator, self.max_iter, self.stop_by_eps, self.decimal_places
        )
        # Verify the result is close to the expected root (ln(2) ≈ 0.693)
        self.assertAlmostEqual(root, math.log(2), places=4)
        # Verify that the method generated iteration steps
        self.assertTrue(len(table) > 0)

    def test_secant_method(self):
        """
        Test the Secant method with a quadratic function (x^2 - 4 = 0).
        Expected root: x = 2.0
        """
        # Test case: Simple quadratic function with root at x = 2
        root, table = self.secant.solve(
            "x**2 - 4",  # Function: x^2 - 4 = 0
            1, 3,  # Initial guesses
            self.eps, self.eps_operator, self.max_iter, self.stop_by_eps, self.decimal_places
        )
        # Verify the result is close to the expected root (2.0)
        self.assertAlmostEqual(root, 2.0, places=4)
        # Verify that the method generated iteration steps
        self.assertTrue(len(table) > 0)

    def test_secant_logarithmic(self):
        """
        Test the Secant method with a logarithmic function (ln(x) - 1 = 0).
        Expected root: x = e ≈ 2.718
        """
        # Test case: Logarithmic function with root at x = e ≈ 2.718
        root, table = self.secant.solve(
            "log(x) - 1",  # Function: ln(x) - 1 = 0
            2, 3,  # Initial guesses
            self.eps, self.eps_operator, self.max_iter, self.stop_by_eps, self.decimal_places
        )
        # Verify the result is close to the expected root (e ≈ 2.718)
        self.assertAlmostEqual(root, math.e, places=4)
        # Verify that the method generated iteration steps
        self.assertTrue(len(table) > 0)


class TestLinearSystemMethods(unittest.TestCase):
    """
    Test cases for linear system numerical methods.
    
    This test class verifies the functionality of all linear system methods:
    - Gauss Elimination Method
    - LU Decomposition Method (if available)
    - Gauss-Jordan Method (if available)
    
    Each test focuses on a specific method with simple test cases that have known solutions.
    """
    def setUp(self):
        """Set up test fixtures before each test method is run."""
        # Initialize the Gauss Elimination method
        self.gauss_elimination = GaussEliminationMethod()
        self.decimal_places = 6  # Decimal places for rounding
        
        # Initialize LU Decomposition method if available
        if has_lu:
            self.lu_decomposition = LUDecompositionMethod()
        
        # Initialize Gauss-Jordan method if available
        if has_gauss_jordan:
            self.gauss_jordan = GaussJordanMethod()

    def test_gauss_elimination_method(self):
        """
        Test the Gauss Elimination method with a simple 2x2 system.
        System: 2x + y = 4, 3x + 4y = 11
        Expected solution: x = 1, y = 2
        """
        # Test case: Simple 2x2 system with solution x = 1, y = 2
        matrix_str = "[[2, 1], [3, 4]]"  # Coefficient matrix A
        vector_str = "[4, 11]"  # Constants vector b
        solution, table = self.gauss_elimination.solve(matrix_str, vector_str, self.decimal_places)
        
        # Verify that a solution was found
        self.assertIsNotNone(solution)
        # Verify that the solution has the correct dimension
        self.assertEqual(len(solution), 2)
        # Verify that the solution values match the expected values
        self.assertAlmostEqual(solution[0], 1.0, places=4)  # x = 1
        self.assertAlmostEqual(solution[1], 2.0, places=4)  # y = 2

    def test_gauss_elimination_3x3(self):
        """
        Test the Gauss Elimination method with a 3x3 system.
        System: 2x + y - z = 1, 5x + 2y + 2z = -4, 3x + y + z = 5
        Expected solution: x = 14, y = -32, z = -5
        """
        # Test case: 3x3 system with solution x = 14, y = -32, z = -5
        matrix_str = "[[2, 1, -1], [5, 2, 2], [3, 1, 1]]"
        vector_str = "[1, -4, 5]"
        solution, table = self.gauss_elimination.solve(matrix_str, vector_str, self.decimal_places)
        
        # Verify that a solution was found
        self.assertIsNotNone(solution)
        # Verify that the solution has the correct dimension
        self.assertEqual(len(solution), 3)
        # Verify that the solution values match the expected values
        self.assertAlmostEqual(solution[0], 14.0, places=4)  # x = 14
        self.assertAlmostEqual(solution[1], -32.0, places=4)  # y = -32
        self.assertAlmostEqual(solution[2], -5.0, places=4)  # z = -5

    def test_gauss_elimination_singular(self):
        """
        Test the Gauss Elimination method with a singular matrix.
        System: x + y = 2, x + y = 3 (inconsistent)
        Expected result: None with an error message.
        """
        # Test case: Singular matrix (no unique solution)
        matrix_str = "[[1, 1], [1, 1]]"
        vector_str = "[2, 3]"
        solution, table = self.gauss_elimination.solve(matrix_str, vector_str, self.decimal_places)
        
        # Verify that no solution is returned for a singular matrix
        self.assertIsNone(solution)
        # Verify that an error message is included in the results
        self.assertTrue(any("Error" in str(item) for item in table) or 
                      any("singular" in str(item).lower() for item in table) or
                      any("zero pivot" in str(item).lower() for item in table))

    @unittest.skipIf(not has_lu, "LU Decomposition method not available")
    def test_lu_decomposition_method(self):
        """
        Test the LU Decomposition method with a simple 2x2 system.
        System: 2x + y = 4, 3x + 4y = 11
        Expected solution: x = 1, y = 2
        """
        # Test case: Simple 2x2 system with solution x = 1, y = 2
        matrix_str = "[[2, 1], [3, 4]]"  # Coefficient matrix A
        vector_str = "[4, 11]"  # Constants vector b
        solution, table = self.lu_decomposition.solve(matrix_str, vector_str, self.decimal_places)
        
        # Verify that a solution was found
        self.assertIsNotNone(solution)
        # Verify that the solution has the correct dimension
        self.assertEqual(len(solution), 2)
        # Verify that the solution values match the expected values
        self.assertAlmostEqual(solution[0], 1.0, places=4)  # x = 1
        self.assertAlmostEqual(solution[1], 2.0, places=4)  # y = 2

    @unittest.skipIf(not has_lu, "LU Decomposition method not available")
    def test_lu_decomposition_3x3(self):
        """
        Test the LU Decomposition method with a 3x3 system.
        System: 2x + y - z = 1, 5x + 2y + 2z = -4, 3x + y + z = 5
        Expected solution: x = 14, y = -32, z = -5
        """
        # Test case: 3x3 system with solution x = 14, y = -32, z = -5
        matrix_str = "[[2, 1, -1], [5, 2, 2], [3, 1, 1]]"
        vector_str = "[1, -4, 5]"
        solution, table = self.lu_decomposition.solve(matrix_str, vector_str, self.decimal_places)
        
        # Verify that a solution was found
        self.assertIsNotNone(solution)
        # Verify that the solution has the correct dimension
        self.assertEqual(len(solution), 3)
        # Verify that the solution values match the expected values
        self.assertAlmostEqual(solution[0], 14.0, places=4)  # x = 14
        self.assertAlmostEqual(solution[1], -32.0, places=4)  # y = -32
        self.assertAlmostEqual(solution[2], -5.0, places=4)  # z = -5

    @unittest.skipIf(not has_gauss_jordan, "Gauss-Jordan method not available")
    def test_gauss_jordan_method(self):
        """
        Test the Gauss-Jordan method with a simple 2x2 system.
        System: 2x + y = 4, 3x + 4y = 11
        Expected solution: x = 1, y = 2
        """
        # Test case: Simple 2x2 system with solution x = 1, y = 2
        matrix_str = "[[2, 1], [3, 4]]"  # Coefficient matrix A
        vector_str = "[4, 11]"  # Constants vector b
        solution, table = self.gauss_jordan.solve(matrix_str, vector_str, self.decimal_places)
        
        # Verify that a solution was found
        self.assertIsNotNone(solution)
        # Verify that the solution has the correct dimension
        self.assertEqual(len(solution), 2)
        # Verify that the solution values match the expected values
        self.assertAlmostEqual(solution[0], 1.0, places=4)  # x = 1
        self.assertAlmostEqual(solution[1], 2.0, places=4)  # y = 2

    @unittest.skipIf(not has_gauss_jordan, "Gauss-Jordan method not available")
    def test_gauss_jordan_3x3(self):
        """
        Test the Gauss-Jordan method with a 3x3 system.
        System: 2x + y - z = 1, 5x + 2y + 2z = -4, 3x + y + z = 5
        Expected solution: x = 14, y = -32, z = -5
        """
        # Test case: 3x3 system with solution x = 14, y = -32, z = -5
        matrix_str = "[[2, 1, -1], [5, 2, 2], [3, 1, 1]]"
        vector_str = "[1, -4, 5]"
        solution, table = self.gauss_jordan.solve(matrix_str, vector_str, self.decimal_places)
        
        # Verify that a solution was found
        self.assertIsNotNone(solution)
        # Verify that the solution has the correct dimension
        self.assertEqual(len(solution), 3)
        # Verify that the solution values match the expected values
        self.assertAlmostEqual(solution[0], 14.0, places=4)  # x = 14
        self.assertAlmostEqual(solution[1], -32.0, places=4)  # y = -32
        self.assertAlmostEqual(solution[2], -5.0, places=4)  # z = -5


if __name__ == '__main__':
    # Run all tests when the script is executed directly
    unittest.main()
