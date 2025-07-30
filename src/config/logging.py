import logging
import logging.config
import os
from datetime import datetime
from typing import Dict, Any
from .settings import get_settings

def setup_logging() -> None:
    """Set up application logging configuration."""
    settings = get_settings()
    
    # Ensure logs directory exists
    os.makedirs(settings.logs_directory, exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = os.path.join(
        settings.logs_directory,
        f"agentic_rag_{datetime.now().strftime('%Y%m%d')}.log"
    )
    
    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
                )
            },
            "simple": {
                "format": "%(asctime)s - %(levelname)s - %(message)s"
            },
            "json": {
                "format": (
                    '{"timestamp": "%(asctime)s", "logger": "%(name)s", '
                    '"level": "%(levelname)s", "file": "%(filename)s", '
                    '"line": %(lineno)d, "function": "%(funcName)s", '
                    '"message": "%(message)s"}'
                )
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": log_filename,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "json",
                "filename": os.path.join(settings.logs_directory, "errors.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "agentic_rag": {
                "level": "DEBUG" if settings.debug else "INFO",
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "chromadb": {
                "level": "WARNING",
                "handlers": ["file"],
                "propagate": False
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"]
        }
    }
    
    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(f"agentic_rag.{name}")


class LoggerMixin:
    """Mixin class to provide logging functionality."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)
