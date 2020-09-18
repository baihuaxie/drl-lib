"""
Utilities for logging, etc.
"""

import logger

def configure_logger(log_path, **kwargs):
    """
    Configure logger
    """
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)