"""
Parallel processing utilities for CS336 assignments.

This module provides generic utilities for parallel processing using both
threading and multiprocessing, with support for progress tracking and error handling.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar, Callable

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')


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