"""
Time formatting utilities for CS336 assignments.

This module provides utilities for formatting and displaying elapsed time
in human-readable formats.
"""

from __future__ import annotations


def format_time(elapsed_time: float) -> str:
    """
    Format elapsed time in seconds to human-readable format.

    Args:
        elapsed_time: Time in seconds

    Returns:
        Formatted time string (e.g., "2 min 30 sec" or "3 hr 2 min")

    Examples:
        >>> format_time(150.5)
        '2 min 30 sec'
        >>> format_time(45.2)
        '0 min 45 sec'
        >>> format_time(3665)
        '1 hr 1 min'
        >>> format_time(10920)
        '3 hr 2 min'
    """
    total_seconds = int(elapsed_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours} hr {minutes} min"
    else:
        return f"{minutes} min {seconds} sec"