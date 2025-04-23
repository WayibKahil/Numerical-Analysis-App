import sys
import traceback
import os
from src.ui.app import NumericalApp
from src.utils.logging_config import configure_logging

__version__ = "1.2.0"

def main():
    """Main application entry point with enhanced error handling."""
    try:
        # Configure logging to suppress warnings
        configure_logging()
        
        # Print startup information
        print(f"Numerical Analysis App v{__version__}")
        print("Starting application...")
        
        # Create application data directory if it doesn't exist
        app_data_dir = os.path.join(os.path.expanduser("~"), ".numerical_analysis_app")
        if not os.path.exists(app_data_dir):
            try:
                os.makedirs(app_data_dir)
                print(f"Created application data directory: {app_data_dir}")
            except Exception as e:
                print(f"Warning: Could not create application data directory: {str(e)}")
        
        # Start the application
        app = NumericalApp()
        app.version = __version__  # Set version in the app
        app.run()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0)
    except ImportError as e:
        print(f"Error: Missing required dependency - {str(e)}")
        if "scipy" in str(e):
            print("SciPy is required for advanced matrix operations like Cramer's Rule.")
            print("Please install it with: pip install scipy")
        else:
            print("Please install all required dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Detailed error information:")
        traceback.print_exc()
        
        # Provide troubleshooting information
        print("\nTroubleshooting steps:")
        print("1. Ensure you have all dependencies installed: pip install -r requirements.txt")
        print("2. Check if you have the correct Python version (3.8 or higher)")
        print("3. Try restarting the application")
        print("4. If the issue persists, please report it with the error details above")
        
        sys.exit(1)

if __name__ == "__main__":
    main()