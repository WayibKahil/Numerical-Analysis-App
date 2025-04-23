import unittest
from src.core.methods import BisectionMethod, FalsePositionMethod, FixedPointMethod, NewtonRaphsonMethod, SecantMethod

class TestNumericalMethods(unittest.TestCase):
    def setUp(self):
        self.bisection = BisectionMethod()
        self.false_position = FalsePositionMethod()
        self.fixed_point = FixedPointMethod()
        self.newton_raphson = NewtonRaphsonMethod()
        self.secant = SecantMethod()
        self.eps = 0.001
        self.max_iter = 50

    def test_bisection(self):
        root, table = self.bisection.solve("x**2 - 4", 0, 3, self.eps, self.max_iter, True)
        self.assertAlmostEqual(root, 2.0, places=3)
        self.assertTrue(len(table) > 0)
        self.assertEqual(table[0]["Error %"], "---")

    def test_false_position(self):
        root, table = self.false_position.solve("x**2 - 4", 0, 3, self.eps, self.max_iter, True)
        self.assertAlmostEqual(root, 2.0, places=3)
        self.assertTrue(len(table) > 0)

    def test_fixed_point(self):
        root, table = self.fixed_point.solve("(x + 4/x)/2", 1, self.eps, self.max_iter, True)
        self.assertAlmostEqual(root, 2.0, places=3)
        self.assertTrue(len(table) > 0)

    def test_newton_raphson(self):
        root, table = self.newton_raphson.solve("x**2 - 4", 1, self.eps, self.max_iter, True)
        self.assertAlmostEqual(root, 2.0, places=3)
        self.assertTrue(len(table) > 0)

    def test_secant(self):
        root, table = self.secant.solve("x**2 - 4", 1, 3, self.eps, self.max_iter, True)
        self.assertAlmostEqual(root, 2.0, places=3)
        self.assertTrue(len(table) > 0)

    def test_invalid_interval(self):
        root, table = self.bisection.solve("x**2 - 4", 3, 4, self.eps, self.max_iter, True)
        self.assertIsNone(root)
        self.assertEqual(table[0]["Error"], "No root in this interval")

if __name__ == "__main__":
    unittest.main()