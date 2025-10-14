# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is CS336 Assignment 1: Basics - a PyTorch-based machine learning assignment focused on implementing fundamental neural network components including tokenizers, attention mechanisms, transformers, and optimizers.

**Project Status**: BPE tokenizer training is complete (`cs336_basics/assignment_1/bpe_tokenizer.py`) with 1/3 training tests passing. Encoding/decoding methods still need implementation for the 22 tokenizer tests. All other neural network components need to be implemented through the adapter pattern in `tests/adapters.py`.

## Development Commands

### Environment Management
- Use `uv` for dependency management and virtual environment (currently using Python 3.13.7)
- Run any Python file: `uv run <python_file_path>`
- The environment will be automatically activated when needed

### Testing
- Run all tests: `uv run pytest` (48 tests available)
- Run specific test file: `uv run pytest tests/test_<component>.py`
- Run with verbose output: `uv run pytest -v`
- Tests use snapshot testing and fixtures located in `tests/fixtures/`
- Test collection: `uv run pytest --collect-only` to see all available tests

### Linting and Code Quality
- Configured with Ruff (line length: 120)
- Line number display enabled in pytest with `-s` flag
- Special lint ignores for `__init__.py` files

### Submission
- Create submission package: `./make_submission.sh`
- Runs tests and creates `cs336-spring2025-assignment-1-submission.zip`

## Code Architecture

### Core Structure
- `cs336_basics/`: Main implementation package
  - `assignment_1/bpe_tokenizer.py`: Complete BPE tokenizer implementation with training (✅ COMPLETE)
  - `training_logger.py`: Reusable training progress logger with timing and stats
  - `utils.py`: File processing utilities with optimized chunking (300MB target chunks at special token boundaries)
  - `constants.py`: Project constants and configuration
  - `pretokenization_example.py`: Example tokenization code
- `tests/`: Comprehensive test suite with adapter pattern (48 tests total)
  - Test categories: data (1), model (13), nn_utils (3), optimizer (2), serialization (1), tokenizer (22), train_bpe (3)
  - BPE training tests: 1/3 passing (test_train_bpe ✅)
- `data/`: Training data directory (✓ populated with required datasets)
  - TinyStoriesV2-GPT4-train.txt (2.2GB)
  - TinyStoriesV2-GPT4-valid.txt (22MB)
  - owt_train.txt (11.9GB)
  - owt_valid.txt (290MB)

### Key Implementation Pattern
The project uses an **adapter pattern** in `tests/adapters.py` where:
- Test functions call adapter functions (e.g., `run_linear`, `run_embedding`)
- Adapter functions should import and use your actual implementations
- **Currently all 593 lines of adapters raise `NotImplementedError`** - these need to be implemented
- Adapters connect the test framework to your actual neural network implementations

### Implementation Status
**Completed:**
- ✅ **BPE Tokenizer Training** (`cs336_basics/assignment_1/bpe_tokenizer.py`):
  - Complete BPE training algorithm with deterministic tie-breaking
  - Memory-efficient multiprocessing with chunking at special token boundaries
  - Automatic chunk sizing (300MB target) for optimal performance
  - Special token handling (atomic, never split)
  - Progress logging with timing breakdowns
  - Test passing: `test_train_bpe` ✅
- ✅ **Training Logger** (`cs336_basics/training_logger.py`):
  - Clean separation of logging from algorithm logic
  - Timer management and progress tracking
  - Configurable verbosity (silent by default)
  - Reusable for any training task
- ✅ **File Processing Utilities** (`cs336_basics/utils.py`):
  - Optimized `find_chunk_boundaries()`: 10MB search window instead of sequential scanning
  - Smart `process_file_chunks_multiprocessing()`: respects special token boundaries, no arbitrary splitting
  - Worker count auto-adjustment (don't spawn more workers than chunks)
- ✅ Data setup (all required datasets downloaded)

**To Implement:**
- **BPE Tokenizer Encoding/Decoding**: Complete encode() and decode() methods for 22 tokenizer tests
- **All adapter functions** in `tests/adapters.py` (lines 32-593):
  - Neural network components: Linear layers, embeddings, SwiGLU, RMSNorm, SiLU
  - Attention mechanisms: Scaled dot-product attention, multi-head attention with/without RoPE
  - RoPE (Rotary Position Embedding) implementation
  - Transformer blocks and complete language model
  - Data utilities: batch sampling, softmax, cross-entropy
  - Optimization: AdamW optimizer, gradient clipping, learning rate scheduling
  - Serialization: model checkpointing and loading

**Known Issues:**
- `test_train_bpe_speed`: Passes functionality but ~2.3s vs 1.5s limit (O(n²) algorithm, needs priority queue optimization)
- `test_train_bpe_special_tokens`: Merge order differs at index 627 (different tie-breaking on larger dataset)

### Test Coverage
- **Model tests (13)**: Core neural network components
- **Tokenizer tests (22)**: Extensive BPE tokenizer validation including memory limits
- **Training BPE tests (3)**: Tokenizer training pipeline
- **Other utilities**: Data loading, optimization, serialization

### Data Setup
**✓ Already completed** - All required datasets are present in the `data/` directory:
- TinyStoriesV2-GPT4-train.txt (2.2GB)
- TinyStoriesV2-GPT4-valid.txt (22MB)
- owt_train.txt (11.9GB)
- owt_valid.txt (290MB)

If you need to re-download:
```sh
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz
```

### Test-Driven Development Workflow
1. Run tests to see current failures: `uv run pytest` (all 48 tests will fail due to NotImplementedError)
2. Choose a component to implement (start with simpler ones like `run_linear`, `run_embedding`)
3. Create the actual implementation in `cs336_basics/` (create new .py files as needed)
4. Update the corresponding adapter function in `tests/adapters.py` to import and call your implementation
5. Re-run tests to verify correctness against snapshots: `uv run pytest tests/test_<component>.py`
6. Repeat for the next component

### Development Strategy
**Recommended implementation order:**
1. **BPE Tokenizer encode/decode**: Complete the tokenizer to pass 22 tokenizer tests
2. **Basic components**: `run_linear`, `run_embedding`, `run_silu`, `run_softmax`
3. **Normalization**: `run_rmsnorm`
4. **Feed-forward**: `run_swiglu`
5. **Attention**: `run_scaled_dot_product_attention`, `run_rope`, then multi-head attention
6. **Transformer**: `run_transformer_block`, `run_transformer_lm`
7. **Utilities**: `run_get_batch`, `run_cross_entropy`, `run_gradient_clipping`
8. **Optimization**: `get_adamw_cls`, `run_get_lr_cosine_schedule`
9. **Serialization**: `run_save_checkpoint`, `run_load_checkpoint`

### BPE Tokenizer Architecture
The BPE tokenizer implementation demonstrates clean architecture principles:

**Key Design Decisions:**
1. **Separation of Concerns**: Logging logic isolated in `TrainingLogger` class
2. **Memory Efficiency**: Chunked processing with multiprocessing, never load entire file
3. **Deterministic Tie-Breaking**: Lexicographic ordering when pair counts are equal
4. **Special Token Handling**: Combined regex pattern matches special tokens first, keeps them atomic
5. **Smart Chunking**:
   - `find_chunk_boundaries()`: Searches 10MB window around estimated positions for nearest special token
   - `process_file_chunks_multiprocessing()`: Never splits beyond provided boundaries
   - Auto-adjusts worker count to match number of chunks
   - Target 300MB per chunk for optimal I/O

**Training Flow:**
```python
# Clean, readable training code
self.logger.log_step("Step 1: Getting initial word counts...", "word_counting")
word_counts = self.get_word_counts(input_path)
self.logger.log_complete("Word counting", "word_counting")

# BPE merging with progress tracking
for i in range(num_merges_needed):
    pair_counts = self._get_pair_counts(word_counts)
    best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))  # Deterministic tie-breaking
    word_counts = self._apply_merge(word_counts, best_pair[0])
    merges.append(best_pair[0])
    self.logger.log_progress(i + 1, num_merges_needed, "merges")

self.logger.log_training_summary("train", "word_counting", "merges", "vocab")
```

### Important Notes
- All implementations should use proper JAX typing annotations (see examples in `tests/adapters.py`)
- Follow the existing code style (120 char line length, Ruff formatting)
- Tests use snapshot comparisons for numerical accuracy - your implementations must match reference outputs exactly
- Memory limits are enforced in tokenizer tests (test_encode_memory_usage, test_encode_iterable_memory_usage)
- Current project version: 1.0.6 (see `pyproject.toml:3`)