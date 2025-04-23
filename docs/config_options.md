# Configuration Options

This document details the various configuration options available in the Numerical Analysis Application.

## User Interface Settings

These settings can be adjusted via the Settings page in the application.

### Calculation Parameters

| Setting | Description | Default Value | Valid Range |
|---------|-------------|---------------|------------|
| Default Decimal Places | Controls the rounding precision for displayed results | 6 | 0-15 |
| Maximum Iterations | Maximum number of iterations before stopping a calculation | 50 | 1-1000 |

### Convergence Parameters

| Setting | Description | Default Value | Valid Range |
|---------|-------------|---------------|------------|
| Error Tolerance (Epsilon) | Convergence threshold for iterative methods | 0.0001 | > 0 |
| Maximum Epsilon Value | Upper limit for the error tolerance | 1.0 | > 0 |
| Stop Condition | Choose when to stop the iterative process | Error Tolerance | Error Tolerance, Maximum Iterations |

## Command Line Options

These options can be specified when launching the application from the command line.


## Application Data

The application stores data in the following locations:

| Data Type | Location | Description |
|-----------|----------|-------------|
| User Settings | `~/.numerical_analysis_app/settings.json` | User preferences and settings |
| Calculation History | `~/.numerical_analysis_app/history.json` | Record of past calculations |
| Exported PDFs | Current working directory | PDF exports of calculation results |

## Advanced Configuration

### Customizing the Theme

The application theme is defined in `src/ui/theme.py`. You can modify the following properties:

```python
LIGHT_MODE = {
    "bg": "#F0F4F8",       # Background color
    "fg": "#DDE4E6",       # Secondary background color
    "text": "#2D3748",     # Text color
    "button": "#4C51BF",   # Button color
    "button_hover": "#2D3748",  # Button hover color
    "accent": "#4C51BF",   # Accent color
    "primary": "#4C51BF",   # Primary color
    "primary_hover": "#3C41AF",  # Primary hover color
    "table_bg": "#FFFFFF", # Table background
    "table_fg": "#2D3748", # Table text
    "table_heading_bg": "#E2E8F0", # Table header background
    "table_heading_fg": "#2D3748",  # Table header text
    "table_odd_row": "#F8FAFC", # Odd row background
    "table_even_row": "#FFFFFF", # Even row background
    "table_hover": "#E2E8F0" # Table hover color
}
```

To add a new theme:

1. Define a new theme dictionary in `ThemeManager`
2. Add it to the `themes` dictionary in the `__init__` method
3. Update the settings UI to include the new theme option

### Application Logging

Logging is configured in `src/utils/logging_config.py`. The default configuration:

- Logs INFO and above to the console
- Logs WARNING and above to a file in `~/.numerical_analysis_app/app.log`
- Rotates log files when they reach 1MB in size
- Keeps up to 3 backup log files

To modify logging behavior:

```python
def configure_logging():
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.expanduser("~"), ".numerical_analysis_app")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "app.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console handler
            RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)  # File handler
        ]
    )
    
    # Suppress warnings from imported libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("sympy").setLevel(logging.WARNING)
```

## Solver Configuration

The Solver class has several configuration options:

| Property | Description | Default Value |
|----------|-------------|---------------|
| `decimal_places` | Number of decimal places for rounding | 6 |
| `max_iter` | Maximum number of iterations | 50 |
| `eps` | Default error tolerance | 0.0001 |
| `max_eps` | Maximum error tolerance | 1.0 |
| `stop_by_eps` | Whether to stop by error tolerance | True |

You can modify these in code:

```python
from src.core.solver import Solver

solver = Solver()
solver.decimal_places = 8  # Increase precision
solver.max_iter = 100      # Allow more iterations
solver.eps = 0.00001       # Tighter error tolerance
```

## Performance Tuning

For improved performance, consider the following options:

2. **Optimize calculation parameters**
   - Increase maximum iterations for complex problems
   - Adjust error tolerance based on required precision
   - Use appropriate methods for different problem types:
     - Matrix methods: Use LU Decomposition for multiple solves
     - Root-finding: Use Newton-Raphson for speed, Bisection for reliability

3. **System recommendations**
   - Modern CPU with multiple cores
   - At least 4GB RAM
   - Updated Python installation (3.8+)
   - Latest package versions 