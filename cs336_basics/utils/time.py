"""
Time formatting utilities for CS336 assignments.

This module provides utilities for formatting and displaying elapsed time
in human-readable formats.
"""

from __future__ import annotations


def format_time(elapsed_time: float) -> str:
    """
    Format elapsed time in seconds to human-readable 'X min Y sec' format.

    Args:
        elapsed_time: Time in seconds

    Returns:
        Formatted time string (e.g., "2 min 30 sec")

    Examples:
        >>> format_time(150.5)
        '2 min 30 sec'
        >>> format_time(45.2)
        '0 min 45 sec'
        >>> format_time(30)
        '0 min 30 sec'
    """
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    return f"{minutes} min {seconds} sec"