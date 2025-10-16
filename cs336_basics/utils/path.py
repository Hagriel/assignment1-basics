"""
Path management utilities for CS336 assignments.

This module provides utilities for finding and managing paths within the project,
including project root detection, data directory management, and import path setup.
"""

from __future__ import annotations

import sys
from pathlib import Path


def get_project_root() -> Path:
    """
    Find the project root directory by looking for pyproject.toml.

    Returns:
        Absolute path to the project root

    Raises:
        FileNotFoundError: If project root cannot be found
    """
    current = Path(__file__).parent

    # Walk up the directory tree looking for pyproject.toml
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent.resolve()

    msg = "Could not find project root (no pyproject.toml found)"
    raise FileNotFoundError(msg)


def add_project_to_path() -> None:
    """
    Add the project root to Python path for imports.

    This allows importing cs336_basics modules from anywhere.
    Should be called before attempting relative imports when running as script.
    """
    try:
        project_root = get_project_root()
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
    except FileNotFoundError:
        # If we can't find project root, try adding parent directories
        current = Path(__file__).parent
        for parent in [current.parent, current.parent.parent]:
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)


def setup_imports() -> bool:
    """
    Setup imports for running modules as scripts or from tests.

    This function handles the common pattern where modules need to work both:
    - When imported from tests (relative imports work)
    - When run as scripts from command line (need path setup)

    Returns:
        True if running as module, False if running as script
    """
    # Check if we're running as a module (relative imports should work)
    try:
        # Try to get the module's __package__ from the calling frame
        frame = sys._getframe(1)
        if frame.f_globals.get("__package__"):
            return True  # Running as module
    except (AttributeError, ValueError):
        # _getframe not available or invalid frame
        pass

    # Running as script, add project to path
    add_project_to_path()
    return False


def get_data_path(filename: str) -> Path:
    """
    Get the path to a data file, working from any execution context.

    This function handles path resolution whether the script is run:
    - From the project root
    - From the cs336_basics directory
    - From assignment directories
    - From tests
    - From any other location

    Args:
        filename: Name of the file in the data directory

    Returns:
        Absolute path to the data file

    Raises:
        FileNotFoundError: If the data file cannot be found

    Examples:
        >>> from cs336_basics.constants import TINYSTORIES_TRAIN
        >>> get_data_path(TINYSTORIES_TRAIN)  # doctest: +SKIP
        PosixPath("/path/to/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")
    """
    # Try to find project root first
    try:
        project_root = get_project_root()
        data_path = project_root / "data" / filename
        if data_path.exists():
            return data_path.resolve()
    except FileNotFoundError:
        pass

    # Get the current file's directory and try different relative paths
    current_dir = Path(__file__).parent

    # Try different relative paths to find the data directory
    possible_paths = [
        current_dir.parent / "data" / filename,  # From cs336_basics/
        current_dir / "data" / filename,         # From project root
        Path("data") / filename,                 # Current directory has data/
        Path("../data") / filename,              # One level up
        Path("../../data") / filename,           # Two levels up
        Path("../../../data") / filename,        # Three levels up (from assignment dirs)
    ]

    for path in possible_paths:
        if path.exists():
            return path.resolve()

    # If not found, raise an error with helpful information
    searched_paths = [str(p) for p in possible_paths]
    msg = f"Could not find data file '{filename}'. Searched in: {searched_paths}"
    raise FileNotFoundError(msg)


def ensure_data_directory() -> Path:
    """
    Ensure the data directory exists and return its path.

    Returns:
        Absolute path to the data directory

    Raises:
        FileNotFoundError: If project root cannot be found
    """
    project_root = get_project_root()
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# Context manager for temporary path additions
class _TempPathContext:
    """Context manager for temporarily adding paths to sys.path."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.added = False

    def __enter__(self) -> None:
        if self.path not in sys.path:
            sys.path.insert(0, self.path)
            self.added = True

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[misc]
        if self.added and self.path in sys.path:
            sys.path.remove(self.path)


def ensure_project_imports() -> None:
    """
    Ensure project imports work from any execution context.

    This function adds the project root to sys.path if not already present,
    allowing clean absolute imports to work regardless of where the script
    is run from.

    Should be called at the top of any module that needs to import from
    cs336_basics when the module might be run as a script.

    Example:
        >>> # At the top of any cs336_basics module:
        >>> import sys
        >>> from pathlib import Path
        >>>
        >>> # Add project root to path
        >>> project_root = Path(__file__).parent.parent.parent  # Adjust as needed
        >>> if str(project_root) not in sys.path:
        ...     sys.path.insert(0, str(project_root))
        >>>
        >>> # Or use this utility (once it's imported):
        >>> from cs336_basics.utils import ensure_project_imports
        >>> ensure_project_imports()
    """
    try:
        project_root = get_project_root()
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
    except FileNotFoundError:
        # Fallback: try adding parent directories based on current file location
        current = Path(__file__).parent
        for parent in [current.parent, current.parent.parent]:
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)


def temp_add_to_path(path: str | Path) -> _TempPathContext:
    """
    Temporarily add a path to sys.path using a context manager.

    Args:
        path: Path to add to sys.path

    Returns:
        Context manager that handles adding/removing the path

    Example:
        >>> with temp_add_to_path("/some/path"):
        ...     # Can import modules from /some/path here
        ...     pass
        # Path is automatically removed after the with block
    """
    return _TempPathContext(str(path))