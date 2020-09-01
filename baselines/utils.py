"""
Utilities for logging, etc.
"""

def configure_logger(log_path, **kwargs):
    """
    Configure logger
    """
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)