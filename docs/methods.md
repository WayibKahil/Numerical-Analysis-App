# Numerical Methods

This document provides details about the numerical methods implemented in the Numerical Analysis Application.

## Root Finding Methods

### Bisection Method

The Bisection Method is a simple and robust technique for finding roots of a continuous function.

**Algorithm:**
1. Start with an interval [a, b] where f(a) and f(b) have opposite signs
2. Calculate the midpoint c = (a + b) / 2
3. Evaluate f(c)
4. If f(c) is close enough to zero, return c as the root
5. If f(c) has the same sign as f(a), update the interval to [c, b]
6. If f(c) has the same sign as f(b), update the interval to [a, c]
7. Repeat steps 2-6 until convergence or maximum iterations

**Advantages:**
- Always converges if initial interval contains a root
- Relatively simple to implement
- Robust against pathological functions

**Disadvantages:**
- Slow convergence rate (linear)
- Requires initial interval with opposite signs

**Application Inputs:**
- Function f(x)
- Lower bound (a)
- Upper bound (b)
- Error tolerance (ε)
- Maximum iterations

### False Position Method (Regula Falsi)

An improvement over the Bisection Method that uses linear interpolation.

**Algorithm:**
1. Start with an interval [a, b] where f(a) and f(b) have opposite signs
2. Calculate c = b - f(b) * (b - a) / (f(b) - f(a))
3. Evaluate f(c)
4. If f(c) is close enough to zero, return c as the root
5. If f(c) has the same sign as f(a), update the interval to [c, b]
6. If f(c) has the same sign as f(b), update the interval to [a, c]
7. Repeat steps 2-6 until convergence or maximum iterations

**Advantages:**
- Faster convergence than Bisection in most cases
- Always converges if initial interval contains a root
- Uses function values to make better approximations

**Disadvantages:**
- Can be slow for certain functions
- Requires initial interval with opposite signs

**Application Inputs:**
- Function f(x)
- Lower bound (a)
- Upper bound (b)
- Error tolerance (ε)
- Maximum iterations

### Newton-Raphson Method

A fast, iterative method that uses function derivatives.

**Algorithm:**
1. Start with an initial guess x₀
2. Calculate xₙ₊₁ = xₙ - f(xₙ) / f'(xₙ)
3. If |xₙ₊₁ - xₙ| or |f(xₙ₊₁)| is less than error tolerance, return xₙ₊₁ as the root
4. Otherwise, repeat steps 2-3 with xₙ = xₙ₊₁ until convergence or maximum iterations

**Advantages:**
- Quadratic convergence (very fast) when close to root
- Precise results in fewer iterations

**Disadvantages:**
- Requires derivative calculation
- May not converge if starting point is poor
- Can diverge or oscillate for certain functions

**Application Inputs:**
- Function f(x)
- Initial guess (x₀)
- Error tolerance (ε)
- Maximum iterations

### Secant Method

Similar to Newton-Raphson but doesn't require derivatives.

**Algorithm:**
1. Start with two initial guesses x₀ and x₁
2. Calculate xₙ₊₁ = xₙ - f(xₙ) * (xₙ - xₙ₋₁) / (f(xₙ) - f(xₙ₋₁))
3. If |xₙ₊₁ - xₙ| or |f(xₙ₊₁)| is less than error tolerance, return xₙ₊₁ as the root
4. Otherwise, repeat steps 2-3 with xₙ₋₁ = xₙ and xₙ = xₙ₊₁ until convergence or maximum iterations

**Advantages:**
- No derivatives required
- Superlinear convergence (faster than Bisection/False Position)
- Useful when derivatives are difficult to calculate

**Disadvantages:**
- Slower than Newton-Raphson
- Requires two initial guesses
- Can diverge with poor initial values

**Application Inputs:**
- Function f(x)
- First initial guess (x₀)
- Second initial guess (x₁)
- Error tolerance (ε)
- Maximum iterations

## Linear System Methods

### Gauss Elimination

A method for solving systems of linear equations.

**Algorithm:**
1. Convert the system to an augmented matrix [A | b]
2. Use elementary row operations to transform the matrix to upper triangular form
3. Solve the system using back substitution

**Advantages:**
- Efficient for moderate-sized systems
- Relatively easy to implement

**Disadvantages:**
- Numerical instability without pivoting
- Less efficient than specialized algorithms for sparse matrices

**Application Inputs:**
- Coefficient matrix A
- Right-hand side vector b

### Gauss Elimination with Partial Pivoting

An improved version of Gauss Elimination with better numerical stability.

**Algorithm:**
1. Convert the system to an augmented matrix [A | b]
2. For each column, find the row with the largest absolute value in that column
3. Swap this row with the current row (partial pivoting)
4. Use elementary row operations to transform the matrix to upper triangular form
5. Solve the system using back substitution

**Advantages:**
- More numerically stable than standard Gauss Elimination
- Handles ill-conditioned systems better

**Disadvantages:**
- Slightly more complex implementation
- Less efficient than specialized algorithms for sparse matrices

**Application Inputs:**
- Coefficient matrix A
- Right-hand side vector b

### Gauss-Jordan Elimination

An extension of Gauss Elimination that transforms the matrix to reduced row echelon form.

**Algorithm:**
1. Convert the system to an augmented matrix [A | b]
2. Use elementary row operations to transform the matrix to reduced row echelon form
3. Read off the solution directly

**Advantages:**
- Provides the matrix inverse as a byproduct
- Solution can be read directly from the transformed matrix

**Disadvantages:**
- More computationally intensive than Gauss Elimination
- Less numerically stable without pivoting

**Application Inputs:**
- Coefficient matrix A
- Right-hand side vector b

### LU Decomposition

Decomposes a matrix into a product of lower and upper triangular matrices.

**Algorithm:**
1. Decompose matrix A into L and U: A = LU
2. Solve Ly = b for y using forward substitution
3. Solve Ux = y for x using back substitution

**Advantages:**
- Efficient for multiple right-hand sides
- Can be used to calculate determinants and inverses
- No need to repeat decomposition for different b vectors

**Disadvantages:**
- More complex implementation
- Requires extra storage for L and U matrices

**Application Inputs:**
- Coefficient matrix A
- Right-hand side vector b

### Cramer's Rule

Uses determinants to solve linear systems.

**Algorithm:**
1. Calculate the determinant of the coefficient matrix A: D = det(A)
2. For each variable xi, replace the ith column of A with b to form matrix Ai
3. Calculate Di = det(Ai)
4. The solution is xi = Di/D for each variable

**Advantages:**
- Direct formula for the solution
- Useful for theoretical analysis

**Disadvantages:**
- Computationally expensive for larger systems
- Numerical instability for larger matrices

**Application Inputs:**
- Coefficient matrix A
- Right-hand side vector b

## Performance Considerations

For each method, the application tracks and displays:
- Number of iterations
- Execution time
- Error convergence
- Function evaluations

When choosing a method, consider:
1. For root finding:
   - Bisection: When reliability is more important than speed
   - Newton-Raphson: When fast convergence is needed and the derivative is easy to compute
   - Secant: When fast convergence is needed but derivatives are difficult

2. For linear systems:
   - Gauss/Gauss-Jordan: For general systems
   - LU Decomposition: When solving multiple systems with the same coefficient matrix
   - Cramer's Rule: For small systems or theoretical purposes 