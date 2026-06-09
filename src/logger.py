import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(
    name: str = "terrasense",
    log_file: str = "logs/terrasense.log",
    level: str = "INFO"
) -> logging.Logger:
    """Sets up a dual handler logger for console and rolling file output.
    
    Args:
        name: Name of the logger.
        log_file: Path to the log file.
        level: Log level (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
        
    Returns:
        A logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # If logger is already configured, don't add duplicate handlers
    if logger.handlers:
        return logger
        
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File Handler (with rotation)
    try:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            # Resolve relative to current working directory or package root
            log_path = Path(os.getcwd()) / log_path
            
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file log handler: {e}. Logging to console only.")
        
    return logger
