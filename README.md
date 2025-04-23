# Numerical Analysis Application 🧮

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-9cf.svg)](https://github.com/TomSchimansky/CustomTkinter)
[![Matplotlib](https://img.shields.io/badge/Plotting-Matplotlib-orange.svg)](https://matplotlib.org/)

A comprehensive Python application for solving numerical analysis problems, featuring multiple root-finding methods and linear system solvers with a modern GUI interface.


## ✨ Features

### 🔍 Root-Finding Methods
- **Bisection Method** - Finds roots by repeatedly bisecting an interval
- **False Position Method** - Linear interpolation between function values
- **Fixed Point Method** - Iterative application of a function until convergence
- **Newton-Raphson Method** - Uses derivative information for faster convergence
- **Secant Method** - Approximates derivatives using finite differences

### 📊 Linear System Solvers
- **Gauss Elimination** - Forward elimination to create an upper triangular matrix, followed by back-substitution
- **Gauss Elimination with Partial Pivoting** - Enhanced stability through row pivoting
- **LU Decomposition** - Factorizes the coefficient matrix into lower and upper triangular matrices
- **LU Decomposition with Partial Pivoting** - Improved numerical stability for LU factorization
- **Gauss-Jordan Method** - Complete elimination to transform the coefficient matrix into the identity matrix
- **Gauss-Jordan Method with Partial Pivoting** - Enhanced stability for the Gauss-Jordan method
- **Cramer's Rule** - Using determinants to solve systems of linear equations

### 🎨 Modern GUI Interface
- Clean and intuitive design with CustomTkinter
- Light theme support
- Real-time results display
- Interactive function plotting

### 🚀 Advanced Capabilities
- Detailed iteration tables
- Step-by-step solution visualization
- Error analysis and convergence details
- Export to PDF
- History tracking
- Customizable settings

## 🆕 New in Version 1.2.0
- Enhanced Cramer's Rule implementation with:
  - Performance optimizations for large matrices
  - Better numerical stability using LU decomposition for determinant calculation
  - Improved matrix visualization with bordered tables
  - Solution quality assessment
  - Performance metrics
- Comprehensive matrix validation with detailed error messages
- Smart method recommendations based on problem characteristics
- Advanced numerical precision for extreme values
- Better error handling and troubleshooting information
- Diagonal dominance and symmetry detection for matrices
- Performance tracking across all numerical methods

## 🆕 New in Version 1.1.0
- Added Cramer's Rule for solving systems of linear equations
- Improved numerical precision for very large and very small values
- Enhanced error handling with more detailed error messages
- Improved singular matrix detection
- Better handling of edge cases in matrix processing
- Fixed minor UI issues
- Performance optimizations

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HosamDyab/NumericalAnalysisApp.git
   cd NumericalAnalysisApp
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. **Launch the application**:
   ```bash
   python main.py
   ```

2. **Select a numerical method** from the dropdown menu

3. **Enter the required parameters**:
   - For root-finding methods: function, bounds/initial values, and convergence criteria
   - For linear systems: coefficient matrix and right-hand side vector

4. **Click "Solve"** to compute the solution

5. **Explore the results**, including iteration details, plots (for root-finding methods), and the final solution

## 🧪 Methods Implementation

### Root-Finding Methods

Each method implements a different approach to finding roots of nonlinear equations:

| Method | Description | Order of Convergence |
|--------|-------------|---------------------|
| Bisection | Divides interval in half at each step | Linear |
| False Position | Uses secant line for better approximation | Linear (faster than bisection) |
| Fixed Point | Iterates function until convergence | Linear or higher (depends on g(x)) |
| Newton-Raphson | Uses derivatives for rapid convergence | Quadratic |
| Secant | Two-point derivative approximation | ~1.62 (superlinear) |

### Linear System Solvers

Our application implements various methods for solving systems of linear equations:

- Direct methods that provide exact solutions (within floating-point precision)
- Options with partial pivoting for enhanced numerical stability
- Clear visualization of the solution process

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- Developed by **Hosam Dyab** & **Hazem Mohamed**
- Based on numerical analysis algorithms and techniques
- Built with Python, CustomTkinter, NumPy, Matplotlib, and SymPy
