# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is CS336 Assignment 1: Basics - a PyTorch-based machine learning assignment focused on implementing fundamental neural network components including tokenizers, attention mechanisms, transformers, and optimizers.

**Project Status**: BPE tokenizer training is complete (`cs336_basics/assignment_1/bpe_tokenizer.py`) with 1/3 training tests passing. Encoding/decoding methods still need implementation for the 22 tokenizer tests. All other neural network components need to be implemented through the adapter pattern in `tests/adapters.py`.

## Development Commands

### Environment Management
- Use `uv` for dependency management and virtual environment (currently using Python 3.14.0)
- Run any Python file: `uv run <python_file_path>`
- The environment will be automatically activated when needed
- **Considering**: Python 3.14t (free-threaded build with disabled GIL) for potential threading benefits

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
  - `utils/`: Utility modules organized by functionality
    - `training_logger.py`: Progress logger with rate tracking and time estimation
    - `cache/`: JSON-based caching system with domain-specific implementations
      - `base.py`: Generic JSONCache base class
      - `word_counts.py`, `pair_counts.py`, `merges.py`, `vocab.py`: BPE-specific caches
      - `checkpoint.py`: CheckpointManager for resumable training (saves every 300 merges)
    - `time.py`: Time formatting with hours/minutes/seconds
    - `file_processing.py`: Optimized chunking (300MB target at special token boundaries)
    - `path.py`: Path resolution utilities
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
  - `cache/{dataset}/`: Cached word counts, pair counts, merges, vocab, and checkpoints

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
  - Sequential merge processing (parallel processing removed due to overhead on large datasets)
  - Progress logging with rate tracking, time estimation, and adaptive intervals
  - Resumable training via checkpoint system (saves every 300 merges)
  - Multi-level caching: word counts, pair counts, merges, vocab, and checkpoints
  - Test passing: `test_train_bpe` ✅
- ✅ **Checkpoint System** (`cs336_basics/utils/cache/checkpoint.py`):
  - `CheckpointManager` class for resumable BPE training
  - Saves incremental checkpoints every 300 merges (reusable across vocab sizes)
  - Auto-loads nearest checkpoint <= target iteration
  - Helper methods: `get_dataset_name()`, `build_vocab()`, `load_or_initialize()`, `save_checkpoint_from_state()`
  - Checkpoint files: `vocab_{iteration}.json`, `merges_{iteration}.json`, `word_counts_{iteration}.json`, `pair_counts_{iteration}.json`
  - Enables safe long-running training on large datasets (OWT ~24-30 hours for 7,935 merges)
- ✅ **Training Logger** (`cs336_basics/utils/training_logger.py`):
  - Clean separation of logging from algorithm logic
  - Timer management and progress tracking with rate calculation (items/sec)
  - Estimated time remaining display
  - Adaptive progress intervals (every 10% or every 10 minutes, whichever is first)
  - Configurable verbosity (silent by default)
  - Reusable for any training task
- ✅ **Cache System** (`cs336_basics/utils/cache/`):
  - Generic `JSONCache` base class with hex-encoded byte serialization
  - Domain-specific caches: `WordCountsCache`, `PairCountsCache`, `MergesCache`, `VocabCache`
  - Hybrid UTF-8/hex encoding for efficient storage
  - Cache directory structure: `data/cache/{dataset}/` with multiple file types
  - Filename-only logging (shows "Saved: filename.json" instead of full path)
- ✅ **File Processing Utilities** (`cs336_basics/utils/file_processing.py`):
  - Optimized `find_chunk_boundaries()`: 10MB search window instead of sequential scanning
  - Smart `process_file_chunks_multiprocessing()`: respects special token boundaries, no arbitrary splitting
  - Worker count auto-adjustment (don't spawn more workers than chunks)
- ✅ **Time Formatting** (`cs336_basics/utils/time.py`):
  - `format_time()` with hours support (e.g., "3 hr 2 min" for durations >= 1 hour)
  - Automatic unit selection (hours + minutes, or minutes + seconds)
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
1. **Separation of Concerns**: Logging logic isolated in `TrainingLogger` class, checkpoint logic in `CheckpointManager`
2. **Memory Efficiency**: Chunked processing with multiprocessing, never load entire file
3. **Deterministic Tie-Breaking**: Lexicographic ordering when pair counts are equal
4. **Special Token Handling**: Combined regex pattern matches special tokens first, keeps them atomic
5. **Smart Chunking**:
   - `find_chunk_boundaries()`: Searches 10MB window around estimated positions for nearest special token
   - `process_file_chunks_multiprocessing()`: Never splits beyond provided boundaries
   - Auto-adjusts worker count to match number of chunks
   - Target 300MB per chunk for optimal I/O
6. **Resumable Training**:
   - Checkpoints saved every 300 merges to `data/cache/{dataset}/`
   - Reusable across different vocab sizes (trains 8192, reuses checkpoint for 4096)
   - Auto-loads nearest checkpoint <= target iteration
   - Enables safe training on large datasets (OWT ~24-30 hours)
7. **Multi-Level Caching**:
   - Word counts cached (avoids reprocessing entire dataset)
   - Pair counts cached (initial computation is expensive)
   - Final merges and vocab cached per vocab_size
   - Checkpoints cached per iteration (incremental progress)
8. **Sequential Merge Processing**:
   - Originally used multiprocessing for merge loop
   - Removed due to overhead on large datasets (6.7M unique words in OWT)
   - Pool creation + serialization overhead > actual computation time
   - Sequential processing is faster and simpler

**Training Flow:**
```python
# Clean, readable training code with checkpoint integration
self.logger.log_step("Step 1: Getting initial word counts...", "word_counting")
word_counts = self.get_word_counts(input_path)
self.logger.log_complete("Word counting", "word_counting")

# Load checkpoint or initialize fresh state
checkpoint_manager = CheckpointManager(dataset_name, logger=self.logger)
checkpoint_state = checkpoint_manager.load_or_initialize(num_merges_needed, self.special_tokens)
merges = checkpoint_state['merges']
start_iteration = checkpoint_state['start_iteration']

# BPE merging with progress tracking and checkpointing
for merge_iteration in range(start_iteration, num_merges_needed):
    pair_counts = self._get_pair_counts(word_counts)
    best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))  # Deterministic tie-breaking
    word_counts = self._apply_merge(word_counts, best_pair[0])
    merges.append(best_pair[0])

    # Adaptive progress logging (every 10% or 10 minutes)
    if is_progress_milestone:
        self.logger.log_progress(merge_iteration + 1, num_merges_needed, "merges")
        # Output: "Progress: 793/7,935 (10%) - 3 hr 2 min elapsed (0.07 per sec, ~27 hr 19 min remaining)"

    # Save checkpoint every 300 merges
    if (merge_iteration + 1) % 300 == 0:
        checkpoint_manager.save_checkpoint_from_state(
            iteration=merge_iteration + 1,
            merges=merges,
            word_counts=word_counts,
            pair_counts=pair_counts,
            special_tokens=self.special_tokens
        )

self.logger.log_training_summary("train", "word_counting", "merges", "vocab")
```

**Cache Directory Structure:**
```
data/cache/{dataset}/
├── word_counts.json              # Initial word frequencies (reused across all vocab sizes)
├── pair_counts.json              # Initial pair frequencies (reused across all vocab sizes)
├── vocab_300.json                # Checkpoint: vocab at 300 merges
├── merges_300.json               # Checkpoint: merges list at 300 merges
├── word_counts_300.json          # Checkpoint: word counts at 300 merges
├── pair_counts_300.json          # Checkpoint: pair counts at 300 merges
├── vocab_600.json                # Checkpoint: vocab at 600 merges
├── merges_600.json               # Checkpoint: merges list at 600 merges
├── ...                           # More checkpoints every 300 merges
├── merges_vocab8192.json         # Final cache: merges for vocab_size=8192
└── vocab_vocab8192.json          # Final cache: vocab for vocab_size=8192
```

### Important Notes
- All implementations should use proper JAX typing annotations (see examples in `tests/adapters.py`)
- Follow the existing code style (120 char line length, Ruff formatting)
- Tests use snapshot comparisons for numerical accuracy - your implementations must match reference outputs exactly
- Memory limits are enforced in tokenizer tests (test_encode_memory_usage, test_encode_iterable_memory_usage)
- Current project version: 1.0.6 (see `pyproject.toml:3`)

### BPE Training Performance Characteristics

**Dataset Statistics:**
- TinyStories (~2.2GB): ~500K unique words, fast training (~minutes)
- OWT (~11.9GB): ~6.7M unique words, slow training (~24-30 hours for 8K vocab)

**Performance Insights:**
- Merge algorithm is O(num_merges × num_unique_words) - not parallelizable
- With 6.7M unique words, multiprocessing overhead exceeds computation time
- Sequential processing is optimal for large datasets
- Checkpoints are critical for long-running training (power outages, experiments, etc.)
- Progress logging adapts to training speed (10% milestones or 10-minute intervals)

**Optimization Attempts:**
- ❌ Multiprocessing for merge loop: Overhead too high (pool creation + serialization)
- ✅ Checkpoint system: Enables resumability and vocab size experimentation
- ✅ Multi-level caching: Word counts, pair counts, merges, vocab
- ✅ Adaptive progress logging: User feedback during slow training

## Python 3.13+ Naming Conventions

Follow these naming best practices for all code in this project:

### **Parameters and Arguments**
- ✅ Use **descriptive, full words**: `pattern` not `pat_str`
- ✅ Use **domain-specific terms**: `left, right` not `piece1, piece2`
- ✅ Be **explicit about types**: `pretokenization_pattern` not `compiled_pat`

### **Instance Attributes**
- ✅ Use **clear intent names**: `self.tokenization_pattern` not `self.combined_pat`
- ✅ Indicate **purpose explicitly**: `self.pretokenization_pattern` vs `self.tokenization_pattern`
- ✅ Avoid **abbreviations**: `self.logger` not `self.log`

### **Loop Variables**
- ❌ **Never use generic `i, j, k`** in complex logic
- ✅ Use **semantic names**:
  - `for merge_iteration in range(num_merges):` not `for i in range(num_merges):`
  - `for position in range(len(word)):` not `for i in range(len(word)):`
  - `for token_idx, token in enumerate(tokens):` not `for idx, token in enumerate(tokens):`
  - `for byte_value in range(256):` not `for i in range(256):`

### **Local Variables**
- ✅ Use **context-specific names**: `chunk_text` not `chunk`
- ✅ Indicate **what was processed**: `matched_text` not `text`
- ✅ Describe **transformations**: `merged_sequence` not `new_word`
- ✅ Use **algorithm-specific terms**: `most_frequent_pair` not `best_pair`

### **Examples from BPE Tokenizer**

**Good naming (after refactoring):**
```python
def __init__(self, pattern: str, special_tokens: list[str], verbose: bool = False):
    self.pretokenization_pattern = re.compile(pattern)
    self.tokenization_pattern = re.compile(f'({special_pattern})|{pattern}')

for merge_iteration in range(num_merges_needed):
    most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
    for pos in range(len(word) - 1):
        if word[pos] == most_frequent_pair[0][0]:
            ...
```

**Bad naming (avoid):**
```python
def __init__(self, pat_str: str, special_tokens: list[str], verbose: bool = False):
    self.compiled_pat = re.compile(pat_str)
    self.combined_pat = re.compile(f'({special_pattern})|{pat_str}')

for i in range(num_merges_needed):
    best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
    for j in range(len(word) - 1):
        if word[j] == best_pair[0][0]:
            ...
```

### **ML/NLP Specific Guidelines**
- Use established terminology: `vocab`, `merges`, `tokens`, `embeddings`
- Be explicit about dimensions: `num_heads`, `hidden_dim`, `vocab_size`
- Clarify intent in transformations: `normalized_weights` not `norm_w`
- Index variables should indicate what they index: `token_idx`, `batch_idx`, `head_idx`