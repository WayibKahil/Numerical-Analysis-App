# User Guide

This guide will walk you through using the Numerical Analysis Application and its features.

## Application Overview

The Numerical Analysis Application provides tools for solving various mathematical problems using numerical methods. The application has a user-friendly interface with several key components:

1. **Sidebar** - Navigation to different sections (Home, History, Settings, About)
2. **Input Form** - Where you enter equations and parameters
3. **Results Table** - Displays calculation steps and iterations
4. **Plot Area** - Visualizes the function and solution
5. **Result Display** - Shows the final solution

## Getting Started

### Launch the Application

Run the application using:

```bash
python main.py
```


### Main Interface

Upon startup, you'll see the welcome screen followed by the main interface:


## Solving Problems

### Step 1: Select a Numerical Method

From the dropdown menu, select one of the available methods:
- Bisection Method
- False Position Method
- Newton-Raphson Method
- Secant Method
- Gauss Elimination
- Gauss-Jordan
- LU Decomposition
- Cramer's Rule

### Step 2: Enter the Function

Enter your mathematical function in the input field. For example:
- `x^2 - 4`
- `sin(x) + cos(x)`
- `e^x - 5`

Use standard mathematical notation. The application supports:
- Basic operations: `+`, `-`, `*`, `/`, `^`
- Functions: `sin`, `cos`, `tan`, `exp`, `log`, etc.
- Constants: `e`, `pi`

### Step 3: Set Parameters

Depending on the selected method, enter the required parameters:
- Initial guesses (x0, x1)
- Error tolerance (epsilon)
- Maximum iterations

### Step 4: Solve

Click the "Solve" button to calculate the solution. The application will:
1. Process your inputs
2. Run the selected numerical method
3. Display the iterations in the results table
4. Plot the function with the root highlighted
5. Show the final result

## Working with Results

### Viewing Iterations

The results table shows each iteration of the solution process:
- For root-finding methods: values at each step, errors, etc.
- For matrix methods: transformation steps

### Interpreting the Plot

The plot shows:
- The function curve
- The root (marked with a red dot)
- Iteration points (if applicable)
- Grid lines for reference

### Exporting Results

To save your solution:
1. Click the "Export to PDF" button
2. The application will generate a PDF file with:
   - The function and method used
   - All iteration steps
   - The final solution
   - A plot of the function (for root-finding methods)

## Using History

The History section keeps track of your calculations:

1. Click "History" in the sidebar
2. View a list of previous calculations
3. Select any entry to view the details
4. Click "View Full Solution" for complete information
5. Use "Clear History" to remove all entries

## Customizing Settings

Customize the application behavior in the Settings section:

1. Click "Settings" in the sidebar
2. Adjust parameters:
   - Default decimal places (affects result precision)
   - Maximum iterations
   - Error tolerance (epsilon)
   - Convergence criteria

3. Click "Save Settings" to apply changes
4. Use "Reset to Defaults" to restore original settings

This mode:
- Skips the welcome animation
- Reduces visual effects
- Minimizes resource usage

## Keyboard Shortcuts

- `Escape` - Cancel current calculation
- `Ctrl+E` - Export current solution to PDF
- `Ctrl+H` - Go to History
- `Ctrl+S` - Go to Settings
- `Ctrl+Home` - Return to Home screen

## Next Steps

For detailed information about the numerical methods implemented in this application, see the [Numerical Methods](methods.md) documentation. 