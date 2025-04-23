# Numerical Analysis App - Usage Guide

This guide provides detailed instructions on how to use the Numerical Analysis App effectively.

## Getting Started

1. Launch the application by running `python main.py`
2. The application will start with a welcome screen and automatically transition to the main interface
3. You can switch between light and dark themes using the theme selector in the top-right corner
4. Toggle full-screen mode using the button next to the theme selector

## Using the Application

### Selecting a Method

The application supports several numerical methods for finding roots of functions:

- **Bisection Method**: Requires an interval [a, b] where the function changes sign
- **False Position Method**: Similar to bisection but uses a weighted average
- **Fixed Point Iteration**: Requires an initial guess and a function g(x) that converges to the root
- **Newton-Raphson Method**: Requires an initial guess and the derivative of the function
- **Secant Method**: Requires two initial guesses and doesn't need the derivative

### Entering a Function

1. Type your function in the "Function f(x):" field using Python syntax
2. You can use the "Select Example" dropdown to choose from predefined functions
3. Supported operations include:
   - Basic arithmetic: +, -, *, /, ** (power)
   - Trigonometric functions: sin(x), cos(x), tan(x)
   - Exponential and logarithmic: exp(x), log(x)
   - Constants: pi, e

### Setting Parameters

Each method requires specific parameters:

- **Bisection/False Position**: Enter the interval bounds (a, b)
- **Fixed Point/Newton**: Enter the initial guess (x0)
- **Secant**: Enter two initial guesses (x0, x1)
- **All Methods**: Set error tolerance and maximum iterations

### Viewing Results

After clicking "Solve", the application will:

1. Display a table showing each iteration
2. Show the final root value
3. Save the result to history

### Exporting Results

Click the "Export to PDF" button to save your results as a PDF file.

## Tips and Tricks

- For better convergence, choose initial values close to the expected root
- If a method fails to converge, try a different method or adjust parameters
- The error tolerance determines the precision of the result
- The history feature allows you to review past calculations

## Troubleshooting

- **Function Syntax Error**: Ensure your function uses valid Python syntax
- **Convergence Issues**: Try different initial values or a different method
- **Application Crashes**: Check the console for error messages 