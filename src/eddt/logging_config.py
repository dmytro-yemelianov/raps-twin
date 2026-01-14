"""Logging configuration utilities."""

import json
import logging
import os
from typing import Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure root logger.

    Args:
        level: Log level string (e.g., "DEBUG", "INFO")
        json_output: Emit JSON logs when True; otherwise plain text
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers (avoid duplicate logs during reload)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)

    root.addHandler(handler)

