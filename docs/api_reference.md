# API Reference

This document provides reference information for developers who want to extend the Numerical Analysis Application. It covers the main classes, methods, and architecture of the application.

## Project Structure

```
NumericalAnalysisApp/
├── main.py                 # Application entry point
├── requirements.txt        # Package dependencies
├── src/                    # Source code
│   ├── core/               # Core numerical algorithms
│   │   ├── methods/        # Implementation of numerical methods
│   │   ├── solver.py       # Main solver class
│   │   └── history.py      # History management
│   ├── ui/                 # User interface components
│   │   ├── app.py          # Main application window
│   │   ├── theme.py        # Theme management
│   │   ├── widgets/        # Custom UI widgets
│   │   └── pages/          # Application pages
│   └── utils/              # Utility functions
│       ├── export.py       # Export functionality
│       └── logging_config.py # Logging configuration
├── docs/                   # Documentation
└── tests/                  # Unit tests
```

## Core Classes

### NumericalApp

The main application class that initializes and manages the UI.

**File:** `src/ui/app.py`

**Key Methods:**
- `__init__()`: Initializes the application
- `run()`: Starts the application
- `solve(**kwargs)`: Solves numerical problems
- `export_solution()`: Exports solutions to PDF
- `show_home()`, `show_history()`, `show_settings()`, `show_about()`: Display different pages

**Example:**
```python
from src.ui.app import NumericalApp

app = NumericalApp()
app.version = "1.2.0"
app.run()
```

### Solver

Responsible for solving numerical problems using various methods.

**File:** `src/core/solver.py`

**Key Methods:**
- `solve(method, f_str, params, eps, eps_operator, max_iter, stop_by_eps, decimal_places)`: Solves a problem
- `bisection(f, a, b, eps, max_iter, stop_by_eps)`: Implements bisection method
- `false_position(f, a, b, eps, max_iter, stop_by_eps)`: Implements false position method
- `newton_raphson(f, x0, eps, max_iter, stop_by_eps)`: Implements Newton-Raphson method
- `secant(f, x0, x1, eps, max_iter, stop_by_eps)`: Implements secant method
- `gauss_elimination(A, b)`: Implements Gauss elimination
- `gauss_jordan(A, b)`: Implements Gauss-Jordan elimination
- `lu_decomposition(A, b)`: Implements LU decomposition
- `cramers_rule(A, b)`: Implements Cramer's rule

**Example:**
```python
from src.core.solver import Solver

solver = Solver()
result, table_data = solver.solve(
    method="Bisection Method",
    f_str="x^2 - 4",
    params={"a": 0, "b": 3},
    eps=0.0001,
    eps_operator="<=",
    max_iter=50,
    stop_by_eps=True,
    decimal_places=6
)
```

### ThemeManager

Manages application themes and appearance.

**File:** `src/ui/theme.py`

**Key Methods:**
- `apply_theme()`: Applies the current theme
- `set_theme(theme_name)`: Sets a new theme

**Example:**
```python
from src.ui.theme import ThemeManager

theme_manager = ThemeManager()
theme = theme_manager.apply_theme()
```

### HistoryManager

Manages calculation history.

**File:** `src/core/history.py`

**Key Methods:**
- `save_solution(func, method, root, table_data)`: Saves a solution to history
- `load_history()`: Loads calculation history
- `clear_history()`: Clears all history

**Example:**
```python
from src.core.history import HistoryManager

history = HistoryManager()
history.save_solution("x^2-4", "Bisection", 2.0, table_data)
history_data = history.load_history()
```

## Extending the Application

### Adding a New Numerical Method

To add a new numerical method:

1. Create a new method module in `src/core/methods/`
2. Implement the method algorithm
3. Add the method to the `Solver` class in `src/core/solver.py`
4. Update the method selection UI in `src/ui/widgets/input_form.py`

**Example Implementation:**

```python
# src/core/methods/new_method.py
import numpy as np

def new_method(f, x0, eps, max_iter):
    """
    Implementation of a new numerical method.
    
    Args:
        f: Function to solve
        x0: Initial guess
        eps: Error tolerance
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (root, iterations_table)
    """
    # Method implementation
    # ...
    
    return root, iterations_table

# In src/core/solver.py
from src.core.methods.new_method import new_method

class Solver:
    # ...
    def __init__(self):
        self.methods = {
            # Existing methods...
            "New Method": self.new_method,
        }
    
    def new_method(self, f_str, params, eps, eps_operator, max_iter, stop_by_eps, decimal_places):
        # Parse parameters
        x0 = float(params.get("x0", 0))
        
        # Create function from string
        f = self._create_function(f_str)
        
        # Call method implementation
        root, table_data = new_method(f, x0, eps, max_iter)
        
        return root, table_data
```

### Customizing the UI

To customize the UI:

1. Modify or extend existing widgets in `src/ui/widgets/`
2. Update theme settings in `src/ui/theme.py`
3. Add new pages or modify existing ones in `src/ui/app.py`

**Example: Adding a Custom Widget:**

```python
# src/ui/widgets/custom_widget.py
import customtkinter as ctk

class CustomWidget:
    def __init__(self, parent, theme, callback):
        self.frame = ctk.CTkFrame(parent, fg_color=theme["bg"])
        
        # Add widget components
        self.label = ctk.CTkLabel(
            self.frame,
            text="Custom Widget",
            font=("Helvetica", 14, "bold"),
            text_color=theme["text"]
        )
        self.label.pack(pady=10)
        
        # Add interaction
        self.button = ctk.CTkButton(
            self.frame,
            text="Action",
            command=callback,
            fg_color=theme["button"],
            hover_color=theme["button_hover"]
        )
        self.button.pack(pady=10)
    
    def update_theme(self, theme):
        """Update widget with new theme."""
        self.frame.configure(fg_color=theme["bg"])
        self.label.configure(text_color=theme["text"])
        self.button.configure(
            fg_color=theme["button"],
            hover_color=theme["button_hover"]
        )
```

### Error Handling

The application uses Python's logging module for error tracking. You should follow this pattern when extending the application:

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    try:
        # Implementation
        pass
    except Exception as e:
        logger.error(f"Error in my_function: {str(e)}")
        # Handle error appropriately
        raise
```

## Command Line Arguments

The application supports command line arguments for various modes:

```python
# main.py
import argparse

parser = argparse.ArgumentParser(description="Numerical Analysis App")
parser.add_argument("--light", action="store_true", help="Run in lightweight mode")
# Add your own arguments here
args = parser.parse_args()
```

To add new command line arguments:
1. Add them to the argument parser in `main.py`
2. Process the arguments in the main function
3. Pass the arguments to the NumericalApp constructor

## Testing and Extension Guidelines

When extending the application:

1. Write unit tests for new functionality in the `tests/` directory
2. Follow existing code style and patterns
3. Document new functions and classes with docstrings
4. Update relevant documentation files
5. Ensure backward compatibility 