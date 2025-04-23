# Tests

This directory contains all the test files for the Numerical Analysis Application.

## Test Structure

- `test_solver.py`: Tests for the core solver functionality
- More test files will be added as the project grows

## Running Tests

To run the tests, you can use the following commands:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src

# Run a specific test file
pytest tests/test_solver.py

# Run tests with verbose output
pytest -v
```

## Test Dependencies

The test dependencies are listed in `requirements-dev.txt`. To install them:

```bash
pip install -r requirements-dev.txt
```

## Code Style

The project uses flake8 for code style checking. The configuration is in `setup.cfg`.

To check code style:

```bash
flake8 src tests
```

## Adding New Tests

When adding new features or fixing bugs, please add corresponding tests. Follow these guidelines:

1. Test files should be named `test_*.py`
2. Test classes should be named `Test*`
3. Test methods should be named `test_*`
4. Use descriptive names that explain what is being tested
5. Include docstrings explaining the test purpose
6. Keep tests focused and independent
7. Use appropriate assertions for the test case 