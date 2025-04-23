import unittest
import sys
import os
import math

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.solver import Solver

class TestSolver(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.solver = Solver()
        
    def test_bisection_method(self):
        """Test the bisection method with various functions."""
        # Test case 1: Simple quadratic function
        func = "x**2 - 4"
        params = {"xl": 0, "xu": 3}
        root, _ = self.solver.solve("Bisection", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, 2.0, places=4)
        
        # Test case 2: Cubic function
        func = "x**3 - 2*x - 5"
        params = {"xl": 2, "xu": 3}
        root, _ = self.solver.solve("Bisection", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, 2.0946, places=4)
        
        # Test case 3: Trigonometric function
        func = "sin(x) - 0.5"
        params = {"xl": 0, "xu": 1.6}
        root, _ = self.solver.solve("Bisection", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, math.pi/6, places=4)
        
    def test_newton_method(self):
        """Test the Newton-Raphson method with various functions."""
        # Test case 1: Simple quadratic function
        func = "x**2 - 4"
        params = {"xi": 3}
        root, _ = self.solver.solve("Newton-Raphson", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, 2.0, places=4)
        
        # Test case 2: Exponential function
        func = "exp(x) - 2"
        params = {"xi": 0}
        root, _ = self.solver.solve("Newton-Raphson", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, math.log(2), places=4)
        
        # Test case 3: Trigonometric function
        func = "cos(x) - x"
        params = {"xi": 0.5}
        root, _ = self.solver.solve("Newton-Raphson", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, 0.7391, places=4)
        
    def test_secant_method(self):
        """Test the secant method with various functions."""
        # Test case 1: Simple quadratic function
        func = "x**2 - 4"
        params = {"xi_minus_1": 1, "xi": 3}
        root, _ = self.solver.solve("Secant", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, 2.0, places=4)
        
        # Test case 2: Cubic function
        func = "x**3 - 2*x - 5"
        params = {"xi_minus_1": 2, "xi": 3}
        root, _ = self.solver.solve("Secant", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, 2.0946, places=4)
        
        # Test case 3: Logarithmic function
        func = "log(x) - 1"
        params = {"xi_minus_1": 2, "xi": 3}
        root, _ = self.solver.solve("Secant", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, math.e, places=4)
        
    def test_fixed_point_method(self):
        """Test the fixed point method with various functions."""
        # Test case 1: Simple quadratic function
        func = "x**2 - 4"
        params = {"xi": 3}
        root, _ = self.solver.solve("Fixed Point", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, 2.0, places=4)
        
        # Test case 2: Trigonometric function
        func = "sin(x) - x/2"
        params = {"xi": 1}
        root, _ = self.solver.solve("Fixed Point", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, 1.8955, places=4)
        
    def test_false_position_method(self):
        """Test the false position method with various functions."""
        # Test case 1: Simple quadratic function
        func = "x**2 - 4"
        params = {"xl": 0, "xu": 3}
        root, _ = self.solver.solve("False Position", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, 2.0, places=4)
        
        # Test case 2: Exponential function
        func = "exp(x) - 3"
        params = {"xl": 0, "xu": 2}
        root, _ = self.solver.solve("False Position", func, params, 0.0001, "<=", 50, True)
        self.assertAlmostEqual(root, math.log(3), places=4)
        
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid function
        func = "invalid_function"
        params = {"xl": 0, "xu": 3}
        root, error = self.solver.solve("Bisection", func, params, 0.0001, "<=", 50, True)
        self.assertIsNone(root)
        self.assertIn("Error", error[0])
        
        # Test invalid interval
        func = "x**2 - 4"
        params = {"xl": 3, "xu": 0}  # Invalid interval
        root, error = self.solver.solve("Bisection", func, params, 0.0001, "<=", 50, True)
        self.assertIsNone(root)
        self.assertIn("Error", error[0])
        
        # Test invalid epsilon
        func = "x**2 - 4"
        params = {"xl": 0, "xu": 3}
        root, error = self.solver.solve("Bisection", func, params, -0.0001, "<=", 50, True)
        self.assertIsNone(root)
        self.assertIn("Error", error[0])

if __name__ == '__main__':
    unittest.main() 