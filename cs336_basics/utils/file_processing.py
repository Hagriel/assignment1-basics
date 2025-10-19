"""
File processing utilities for CS336 assignments.

This module provides utilities for processing large files efficiently,
including file chunking at special token boundaries and multiprocessing
for parallel file processing.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, TypeVar, Callable

if TYPE_CHECKING:
    from typing import BinaryIO

# Type variable for generic functions
R = TypeVar('R')


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

    for boundary_idx in range(1, len(chunk_boundaries) - 1):
        estimated_position = chunk_boundaries[boundary_idx]

        # Read a window around the estimated position
        search_start = max(0, estimated_position - search_window // 2)
        search_end = min(file_size, estimated_position + search_window // 2)

        file.seek(search_start)
        window = file.read(search_end - search_start)

        if not window:
            chunk_boundaries[boundary_idx] = file_size
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
            chunk_boundaries[boundary_idx] = closest
        else:
            # No special token found in window, keep estimated position
            # (This is a fallback - ideally special tokens should be common enough)
            chunk_boundaries[boundary_idx] = estimated_position

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


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
    recommended_max_chunk_size: int = 300 * 1024 * 1024,  # 300MB recommendation (not enforced)
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