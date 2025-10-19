# BPE Tokenizer Implementation - Architecture Overview

## Current Architecture (January 2025)

### File Structure

```
cs336_basics/assignment_1/
├── tokenizer.py                    # Base tokenizer (encoding/decoding)
├── bpe_tokenizer.py               # BPE training (inherits from Tokenizer)
└── tokenizer_data_manager.py     # Data loading and caching

cs336_basics/utils/
├── training_logger.py             # Progress logging
├── cache/
│   ├── checkpoint.py              # Resumable training checkpoints
│   ├── word_counts.py             # Word count caching
│   ├── pair_counts.py             # Pair count caching
│   ├── merges.py                  # Merge operations caching
│   └── vocab.py                   # Vocabulary caching
└── file_processing.py             # Chunked file processing
```

### Class Hierarchy

```
Tokenizer (base class)
├── encode(text) -> token_ids
├── decode(token_ids) -> text
├── load_vocab()
└── init_vocab()
        ▲
        │ inherits
        │
BPETokenizer (training class)
├── train(input_path, vocab_size)
├── _initialize_training_state()
├── _run_merge_loop()
├── _get_word_counts()
└── _get_pair_counts()
```

## Key Design Decisions

### 1. Separation of Concerns

**Base Tokenizer** (`tokenizer.py`):
- Pure encoding/decoding functionality
- Vocabulary management
- No training dependencies
- Lightweight for inference

**BPE Tokenizer** (`bpe_tokenizer.py`):
- Training functionality only
- Inherits encoding/decoding from base
- Manages training state and checkpoints

**Data Manager** (`tokenizer_data_manager.py`):
- Single source of truth for vocab building
- Handles all cache operations
- Returns immutable `VocabState` objects

### 2. Multi-Level Caching

```
Cache Hierarchy:
1. Word counts cache       → Avoids reprocessing dataset
2. Pair counts cache       → Avoids expensive initial computation
3. Checkpoint cache        → Every 300 merges (resumable training)
4. Final vocab/merges      → Per vocab_size (instant loading)
```

**Benefits**:
- OWT training: ~24-30 hours → resumable via checkpoints
- Experiments: Reuse checkpoints across vocab sizes
- Re-runs: Instant load from final cache

### 3. Memory-Efficient Training

**Chunked Processing**:
- 300MB chunks (optimal I/O)
- Smart boundaries at special tokens (no mid-word splits)
- Multiprocessing for initial word counting
- Sequential merge processing (6.7M words in OWT)

**Why Sequential Merges?**
- Pool creation overhead > computation time on large datasets
- Simpler and faster for OWT (6.7M unique words)

### 4. Progress Tracking

**Adaptive Logging**:
- Every 10% milestone OR every 10 minutes (whichever first)
- Shows: current speed, average speed, time elapsed, ETA
- Correct speed calculation on checkpoint resume

### 5. Code Reusability

**No Duplication**:
- Vocab building: 3 implementations → 1 (in `TokenizerDataManager`)
- `BPETokenizer.init_vocab()` → delegates to data manager
- `CheckpointManager.build_vocab()` → delegates to data manager

**Single Source of Truth**: Changes propagate automatically

## Usage Examples

### Training from Scratch

```python
from cs336_basics.assignment_1 import BPETokenizer
from cs336_basics.constants import GPT2_PAT_STR

# Create tokenizer
bpe = BPETokenizer(GPT2_PAT_STR, ['<|endoftext|>'], verbose=True)

# Train (with automatic caching and checkpointing)
result = bpe.train(input_path="data/owt_train.txt", vocab_size=8192)

# Use trained tokenizer
tokens = bpe.encode("Hello world!")
text = bpe.decode(tokens)
```

### Loading Pre-trained Vocabulary

```python
from cs336_basics.assignment_1 import Tokenizer

# Lightweight tokenizer for inference
tokenizer = Tokenizer(GPT2_PAT_STR, ['<|endoftext|>'])

# Load from cache
tokenizer.load_vocab(dataset_name="owt_train.txt", vocab_size=8192)

# Encode/decode
tokens = tokenizer.encode("Hello world!")
text = tokenizer.decode(tokens)
```

### Resuming Training

Training automatically resumes from the nearest checkpoint:

```python
# First run: trains 0-300 merges, saves checkpoint
bpe.train("data/owt_train.txt", vocab_size=8192)

# Interrupted, restart
# Second run: loads checkpoint, continues from merge 300
bpe.train("data/owt_train.txt", vocab_size=8192)
```

## Implementation Details

### Vocabulary Structure

```python
# Token ID ranges:
# [0..n-1]        : special tokens (e.g., '<|endoftext|>')
# [n..n+255]      : base bytes (0x00-0xFF)
# [n+256..vocab_size-1] : merged tokens from BPE
```

Example with 1 special token, vocab_size=8192:
- ID 0: `<|endoftext|>` (special token)
- IDs 1-256: Individual bytes
- IDs 257-8191: 7,935 merged tokens from training

### BPE Training Algorithm

```python
# 1. Get word counts (with multiprocessing)
word_counts = get_word_counts(input_path)

# 2. Load checkpoint or initialize
checkpoint = load_nearest_checkpoint(num_merges_needed)

# 3. Merge loop (deterministic tie-breaking)
for merge_iteration in range(start, num_merges_needed):
    # Find most frequent pair (lexicographic tie-break)
    most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))

    # Apply merge to all words
    word_counts = apply_merge(word_counts, most_frequent_pair)

    # Update pair counts incrementally
    pair_counts = update_pair_counts(pair_counts, word_counts)

    # Save checkpoint every 300 merges
    if (merge_iteration + 1) % 300 == 0:
        save_checkpoint(...)
```

### Encoding Algorithm

```python
def encode(text: str) -> list[int]:
    token_ids = []

    # 1. Pre-tokenize using regex pattern
    for match in tokenization_pattern.finditer(text):
        word = match.group()

        # 2. Handle special tokens
        if word in special_tokens:
            token_ids.append(special_token_id)
            continue

        # 3. Convert to byte sequence
        word = [bytes([b]) for b in word.encode('utf-8')]

        # 4. Apply BPE merges (in learned order)
        for merge_pair in merges:
            word = apply_merge(word, merge_pair)

        # 5. Map to token IDs
        for token_bytes in word:
            token_ids.append(bytes_to_id[token_bytes])

    return token_ids
```

## Performance Characteristics

### Dataset Sizes
- **TinyStories** (~2.2GB): ~500K unique words, training ~minutes
- **OWT** (~11.9GB): ~6.7M unique words, training ~24-30 hours for 8K vocab

### Cache Impact
| Operation | Without Cache | With Cache | Speedup |
|-----------|--------------|------------|---------|
| Word counting | 10-15 min | < 1 sec | 600-900x |
| Pair counting | 5-10 min | < 1 sec | 300-600x |
| Load vocab | N/A (train) | < 1 sec | Instant |
| Resume training | Start over | Continue | ~hours saved |

### Memory Usage
- **Chunked processing**: Never loads entire file (handles 12GB OWT)
- **Checkpoint size**: ~50-100MB per checkpoint (depends on unique words)
- **Final cache**: ~10-20MB (vocab + merges)

## Best Practices

### For Training

1. **Use caching**: Leave `use_cache=True` (default)
2. **Enable checkpoints**: Automatic every 300 merges
3. **Monitor progress**: Set `verbose=True` for long training runs
4. **Experiment safely**: Checkpoints are reusable across vocab sizes

### For Inference

1. **Use base Tokenizer**: No training overhead
2. **Load from cache**: Much faster than training
3. **Reuse instances**: Vocab loading is one-time cost

### For Development

1. **Test on TinyStories**: Fast iteration (~minutes)
2. **Scale to OWT**: Production testing (~hours)
3. **Check checkpoints**: Resume capability important for OWT

## Known Limitations

### Performance
- **Encoding**: O(n*m) where n=word_length, m=num_merges
  - Current: ~30K merges max
  - Optimization: Priority queue would give O(n log n)
  - Impact: 100-1000x speedup for encoding

### Algorithm
- **Sequential merges**: Not parallelizable by design
- **Tie-breaking**: Lexicographic (deterministic but arbitrary)
- **Memory**: Entire word_counts dict in memory during training

## Testing Status

- ✅ **test_train_bpe**: Basic training functionality
- ⚠️ **test_train_bpe_speed**: Passes but ~2.3s vs 1.5s limit
- ⚠️ **test_train_bpe_special_tokens**: Merge order differs at index 627

## Future Optimizations

### High Priority
1. Optimize `_apply_bpe_merges()` with priority queue (O(n log n))
2. Add input validation (empty strings, None values)
3. Add batch encoding/decoding methods

### Medium Priority
4. Add streaming encode/decode for large files
5. Add vocab pruning/extending methods
6. Replace print statements with proper logging

### Low Priority
7. Add serialize/deserialize methods for model saving
8. Add token frequency analysis tools
9. Add vocab visualization utilities

## References

### Key Files
- Implementation: `cs336_basics/assignment_1/`
- Tests: `tests/test_tokenizer.py`, `tests/test_train_bpe.py`
- Constants: `cs336_basics/constants.py`

### Algorithms
- BPE: Sennrich et al. (2016) - "Neural Machine Translation of Rare Words with Subword Units"
- Tie-breaking: Lexicographic ordering for determinism
- Checkpointing: Every 300 merges (empirical optimum)

---

**Last Updated**: January 2025
**Version**: 1.0 (Post class-separation refactoring)