"""
Import utilities for CS336 assignments.

This module provides a simple pattern for ensuring imports work
from any execution context. It's designed to be minimal and
have no dependencies on other cs336_basics modules.
"""

import sys
from pathlib import Path


def setup_imports_from_file(file_path: str, levels_up: int = 3) -> None:
    """
    Setup imports for a file by adding the project root to sys.path.

    Args:
        file_path: The __file__ path of the calling module
        levels_up: How many levels up to go to reach project root (default: 3)
                  - For files in cs336_basics/assignment_1/: use 3 (default)
                  - For files in cs336_basics/: use 2
                  - For files in project root: use 1

    Example:
        >>> # At the top of any cs336_basics module:
        >>> from cs336_basics.import_utils import setup_imports_from_file
        >>> setup_imports_from_file(__file__)
        >>>
        >>> # Then use clean imports
        >>> from cs336_basics.utils import format_time
        >>> from cs336_basics.constants import TINYSTORIES_TRAIN
    """
    current_dir = Path(file_path).parent
    project_root = current_dir

    # Go up the specified number of levels
    for _ in range(levels_up - 1):
        project_root = project_root.parent

    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)