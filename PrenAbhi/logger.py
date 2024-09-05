import logging


class LogColors:
    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    PURPLE = "\033[35m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD_RED = "\033[1;31m"


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_colors = {
            logging.DEBUG: LogColors.CYAN,
            logging.INFO: LogColors.GREEN,
            TELEMETRY_LEVEL: LogColors.PURPLE,
            logging.WARNING: LogColors.YELLOW,
            logging.ERROR: LogColors.RED,
            logging.CRITICAL: LogColors.BOLD_RED,
        }
        color = log_colors.get(record.levelno, LogColors.RESET)
        record.levelname = f"{color}{record.levelname}{LogColors.RESET}"

        return super().format(record)


TELEMETRY_LEVEL = 25
logging.addLevelName(TELEMETRY_LEVEL, "TELEMETRY")


def telemetry(self: logging.Logger, message: str, *args, **kws) -> None:
    """
    Custom logging method for telemetry data.

    Args:
        self (logging.Logger): Logger instance.
        message (str): The telemetry message.
        *args: Additional arguments.
        **kws: Additional keyword arguments.
    """
    if self.isEnabledFor(TELEMETRY_LEVEL):
        self._log(TELEMETRY_LEVEL, message, args, **kws)


logging.Logger.telemetry = telemetry


def setup_logger() -> logging.Logger:
    """
    Set up and configure the logger with color formatting.

    Returns:
        logging.Logger: Configured logger instance.
    """
    formatter = ColoredFormatter(
        "%(levelname)s - [%(asctime)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler("application.log")
    file_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s - [%(asctime)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    logger = logging.getLogger(__name__)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


logger = setup_logger()
