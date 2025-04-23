import logging

def configure_logging():
    """Configure the application logging system."""
    # Create a root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)  # Only show ERROR and CRITICAL messages
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    
    # Create a formatter
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add the handler to the root logger
    root_logger.addHandler(console_handler)
    
    # Remove any existing handlers to avoid duplicate messages
    for handler in root_logger.handlers[:-1]:
        root_logger.removeHandler(handler)
