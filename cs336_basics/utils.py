"""
Utility functions for CS336 assignments.

This module contains common utility functions that are used across multiple
assignments and components, including timing, file handling, path resolution,
and import management.

Requires Python 3.13+
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, TypeVar, Any
import multiprocessing as mp

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

if TYPE_CHECKING:
    from typing import BinaryIO


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


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int | None = None,
    split_special_token: bytes = b'<|endoftext|>',
    target_chunk_size: int | None = None,
) -> list[int]:
    """
    Chunk a file into parts that can be processed independently.

    This function splits a large file into approximately equal chunks while
    ensuring that chunks end at natural boundaries (special tokens) to avoid
    splitting tokens in the middle.

    Args:
        file: Open binary file handle to chunk
        desired_num_chunks: Target number of chunks to create (optional if target_chunk_size provided)
        split_special_token: Byte sequence to use as chunk boundary marker (default: b'<|endoftext|>')
        target_chunk_size: Target size per chunk in bytes (e.g., 300 * 1024 * 1024 for 300MB).
                          If provided, desired_num_chunks is auto-calculated.

    Returns:
        List of byte positions marking chunk boundaries.
        May return fewer chunks if boundaries overlap.

    Raises:
        ValueError: If neither desired_num_chunks nor target_chunk_size is provided

    Notes:
        - The returned list will have N+1 elements (start/end positions)
        - First element is always 0, last element is file size
        - Chunks are defined as [boundaries[i], boundaries[i+1])
        - Function may return fewer chunks than requested if file is small
        - Actual chunk sizes will vary based on special token positions

    Example:
        >>> # Auto-calculate chunks for ~300MB each
        >>> with open("data.txt", "rb") as f:
        ...     boundaries = find_chunk_boundaries(f, target_chunk_size=300*1024*1024)
        >>>
        >>> # Or specify exact number of chunks
        >>> with open("data.txt", "rb") as f:
        ...     boundaries = find_chunk_boundaries(f, desired_num_chunks=4)
    """
    if not isinstance(split_special_token, bytes):
        msg = "Must represent special token as a bytestring"
        raise TypeError(msg)

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # Calculate desired_num_chunks from target_chunk_size if provided
    if target_chunk_size is not None:
        if target_chunk_size <= 0:
            msg = "target_chunk_size must be positive"
            raise ValueError(msg)
        desired_num_chunks = max(1, (file_size + target_chunk_size - 1) // target_chunk_size)
    elif desired_num_chunks is None:
        msg = "Must provide either desired_num_chunks or target_chunk_size"
        raise ValueError(msg)

    if desired_num_chunks <= 0:
        msg = "desired_num_chunks must be positive"
        raise ValueError(msg)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    # Search window size: look this far before/after estimated boundary for special token
    search_window = 10 * 1024 * 1024  # 10MB window

    for bi in range(1, len(chunk_boundaries) - 1):
        estimated_position = chunk_boundaries[bi]

        # Read a window around the estimated position
        search_start = max(0, estimated_position - search_window // 2)
        search_end = min(file_size, estimated_position + search_window // 2)

        file.seek(search_start)
        window = file.read(search_end - search_start)

        if not window:
            chunk_boundaries[bi] = file_size
            continue

        # Find all occurrences of special token in the window
        occurrences = []
        start_idx = 0
        while True:
            idx = window.find(split_special_token, start_idx)
            if idx == -1:
                break
            occurrences.append(search_start + idx)
            start_idx = idx + 1

        if occurrences:
            # Find the occurrence closest to our estimated position
            closest = min(occurrences, key=lambda pos: abs(pos - estimated_position))
            chunk_boundaries[bi] = closest
        else:
            # No special token found in window, keep estimated position
            # (This is a fallback - ideally special tokens should be common enough)
            chunk_boundaries[bi] = estimated_position

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


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


def process_chunks_parallel(
    tasks: list[T],
    worker_function: Callable[[T], R],
    num_workers: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
    error_callback: Callable[[Exception, T], None] | None = None,
) -> list[R]:
    """
    Process a list of tasks in parallel using ThreadPoolExecutor.

    This is a generic utility for parallel processing that can be reused
    across different components that need multithreading.

    Args:
        tasks: List of tasks to process
        worker_function: Function that processes a single task
        num_workers: Number of worker threads (default: 4)
        progress_callback: Optional callback for progress updates (completed, total)
        error_callback: Optional callback for handling errors (exception, task)

    Returns:
        List of results in the same order as input tasks

    Example:
        >>> def process_file_chunk(chunk_info):
        ...     start, end, filename = chunk_info
        ...     # Process chunk and return result
        ...     return result
        >>>
        >>> chunks = [(0, 1000, "file.txt"), (1000, 2000, "file.txt")]
        >>> results = process_chunks_parallel(chunks, process_file_chunk, num_workers=2)
    """
    if not tasks:
        return []

    results = [None] * len(tasks)  # Pre-allocate results list
    completed_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks with their original indices
        future_to_index = {
            executor.submit(worker_function, task): i
            for i, task in enumerate(tasks)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            task = tasks[index]

            try:
                result = future.result()
                results[index] = result
                completed_count += 1

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed_count, len(tasks))

            except Exception as e:
                completed_count += 1
                if error_callback:
                    error_callback(e, task)
                else:
                    # Re-raise if no error callback provided
                    raise

    return results


# Global worker function for multiprocessing (must be at module level for pickling)
def _process_file_chunk_worker(args: tuple[str, int, int, Callable[[str, int, int], R]]) -> R:
    """
    Worker function for multiprocessing file chunk processing.

    This function must be at module level to be pickable by multiprocessing.
    """
    file_path, start, end, chunk_processor = args
    return chunk_processor(file_path, start, end)


def process_file_chunks_multiprocessing(
    file_path: str | Path,
    boundaries: list[int],
    chunk_processor: Callable[[str, int, int], R],
    num_workers: int | None = None,
    show_progress: bool = True,
    recommended_max_chunk_size: int = 100 * 1024 * 1024,  # 100MB recommendation (not enforced)
) -> list[R]:
    """
    Process file chunks using multiprocessing for CPU-bound tasks.

    This uses actual separate processes instead of threads, bypassing Python's GIL
    and allowing true parallel processing for CPU-intensive tasks like tokenization.

    Args:
        file_path: Path to the file to process
        boundaries: List of byte positions marking chunk boundaries (e.g., at special tokens)
        chunk_processor: Function that processes (file_path, start, end) -> result
        num_workers: Number of worker processes (default: CPU count)
        show_progress: Whether to show progress updates (default: True)
        recommended_max_chunk_size: Recommended max chunk size (chunks can exceed this)

    Returns:
        List of results from processing each chunk

    Performance Notes:
        - Uses separate processes for true parallelism
        - Better for CPU-bound tasks (like regex processing)
        - Higher memory usage per process
        - Process startup overhead
        - Chunks are NOT split beyond the provided boundaries to maintain consistency

    Example:
        >>> results = process_file_chunks_multiprocessing(
        ...     "large_file.txt", boundaries, process_chunk, num_workers=4
        ... )
    """
    if len(boundaries) < 2:
        return []

    # Create chunk tasks as (start, end) pairs from boundaries
    # Do NOT split chunks further - boundaries are already at special token positions
    chunk_pairs = list(zip(boundaries[:-1], boundaries[1:]))

    # Default to CPU count, but don't use more workers than chunks
    if num_workers is None:
        num_workers = mp.cpu_count()
    num_workers = min(num_workers, len(chunk_pairs))

    if show_progress:
        print(f"Processing {len(chunk_pairs)} chunks using {num_workers} processes (multiprocessing)...")
        # Warn if any chunks are very large
        max_chunk = max(end - start for start, end in chunk_pairs)
        if max_chunk > recommended_max_chunk_size:
            print(f"Warning: Largest chunk is {max_chunk / (1024*1024):.1f}MB (recommended max: {recommended_max_chunk_size / (1024*1024):.1f}MB)")
            print(f"Consider increasing num_workers in find_chunk_boundaries for better distribution")

    # Create task arguments that include the processor function
    tasks = [
        (str(file_path), start, end, chunk_processor)
        for start, end in chunk_pairs
    ]

    # Use ProcessPoolExecutor for true parallelism
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(_process_file_chunk_worker, task): i
            for i, task in enumerate(tasks)
        }

        # Pre-allocate results list
        results = [None] * len(chunk_pairs)

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                result = future.result()
                results[chunk_idx] = result
                completed += 1

                if show_progress and completed % max(1, len(chunk_pairs) // 10) == 0:
                    progress = (completed / len(chunk_pairs)) * 100
                    print(f"Progress: {completed}/{len(chunk_pairs)} chunks ({progress:.1f}%)")

            except Exception as e:
                print(f"Error processing chunk {chunk_idx}: {e}")
                completed += 1

    return results