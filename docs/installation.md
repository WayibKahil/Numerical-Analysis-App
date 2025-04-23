# Installation Guide

This guide will help you install and set up the Numerical Analysis Application on your system.

## System Requirements

- Python 3.8 or higher
- Operating System: Windows, macOS, or Linux
- At least 4GB of RAM (8GB recommended for larger calculations)
- 100MB of disk space

## Installation Steps

### 1. Clone or Download the Repository

```bash
git clone https://github.com/HosamDyab/NumericalAnalysisApp.git
cd NumericalAnalysisApp
```

Or download and extract the ZIP file from the project repository.

### 2. Set Up a Virtual Environment (Recommended)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- customtkinter==5.2.1
- sympy==1.12
- numpy==1.24.3
- matplotlib==3.7.1
- reportlab==4.0.4
- pillow==10.0.0
- scipy==1.11.1

### 4. Verify Installation

Run the application to verify everything is working correctly:

```bash
python main.py
```

You should see the application launch with the welcome screen.

## Troubleshooting Installation

### Missing Dependencies

If you encounter an error about missing dependencies, try installing them individually:

```bash
pip install customtkinter sympy numpy matplotlib reportlab pillow scipy
```

### Python Version Issues

Ensure you're using Python 3.8 or higher:

```bash
python --version
```

If needed, download the appropriate Python version from [python.org](https://www.python.org/downloads/).

### Display Issues


### For Developers

If you're planning to develop or modify the application, also install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Next Steps

Once installed, proceed to the [User Guide](user_guide.md) to learn how to use the application. 