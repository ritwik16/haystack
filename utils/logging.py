import logging
import sys
from logging.handlers import RotatingFileHandler
from utils.config import settings


def setup_logging():
    """Configure logging for the application"""
    # Create logs directory if it doesn't exist
    log_dir = settings.BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Log format
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File handler
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()


def log_qa_to_file(query: str, response: str):
    """Log question and answer to a file"""
    try:
        with open(settings.QA_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"Question: {query}\n")
            f.write(f"Answer: {response}\n")
            f.write("-" * 50 + "\n")
        logger.info(f"QA pair logged: '{query}'")
    except Exception as e:
        logger.error(f"Error writing to QA log file: {str(e)}")