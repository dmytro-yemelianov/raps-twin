"""Utility functions for EDDT."""

from datetime import datetime, time
from typing import Optional


def parse_time(time_str: str) -> time:
    """
    Parse time string (HH:MM) to time object.

    Args:
        time_str: Time string in HH:MM format

    Returns:
        time object
    """
    hour, minute = map(int, time_str.split(":"))
    return time(hour, minute)


def is_within_working_hours(
    check_time: datetime,
    start_time: time = time(8, 0),
    end_time: time = time(17, 0),
) -> bool:
    """
    Check if datetime is within working hours.

    Args:
        check_time: Datetime to check
        start_time: Work start time
        end_time: Work end time

    Returns:
        True if within working hours
    """
    current_time = check_time.time()
    return start_time <= current_time < end_time


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2h 30m", "45m", "30s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        if minutes > 0:
            return f"{hours}h {minutes}m"
        return f"{hours}h"
