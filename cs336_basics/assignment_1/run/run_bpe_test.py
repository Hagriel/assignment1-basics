"""Quick test script for BPE tokenizer training.

This script tests basic BPE tokenizer functionality by training on TinyStories validation data.
"""

from cs336_basics.assignment_1.bpe_tokenizer import BPETokenizer
from cs336_basics.constants import GPT2_PAT_STR, MODEL_DEFAULTS, TINYSTORIES_VALID, OWT_TRAIN

from cs336_basics.assignment_1.tokenizer import Tokenizer
from cs336_basics.constants import GPT2_PAT_STR, MODEL_DEFAULTS, TINYSTORIES_VALID


def test_train_bpe():
    bpe = BPETokenizer(GPT2_PAT_STR, MODEL_DEFAULTS.DEFAULT_SPECIAL_TOKENS, verbose=True)
    train_result = bpe.train(OWT_TRAIN, 32_000)
    print('vocab size = ', len(train_result.vocab))
    print('merges count = ', len(train_result.merges))
    print('First 30 longest merges:', sorted(train_result.merges, key=lambda x: len(x[0] + x[1]), reverse=True)[:30])


def test_bpe():
    """Test encode/decode functionality with BPE tokenizer."""

    print("=" * 80)
    print("BPE Tokenizer Encode/Decode Test")
    print("=" * 80)
    dataset_name = OWT_TRAIN
    vocab_size = 32_000

    # Initialize and train tokenizer
    print(f"\n1. Training tokenizer on {dataset_name} (vocab_size={vocab_size})...")
    bpe = Tokenizer(GPT2_PAT_STR, MODEL_DEFAULTS.DEFAULT_SPECIAL_TOKENS, verbose=True)
    bpe.load_vocab(dataset_name=dataset_name, vocab_size=vocab_size)

    # Test cases
    test_texts = [
        "Hello, world!",
        "I have a pussy you have a dick",
        "The quick brown fox jumps over the lazy dog.",
        "BPE tokenization is fun! 🎉",
        "Special token: <|endoftext|> should work.",
        "",  # Empty string
        "a",  # Single character
        "   ",  # Whitespace
        "Hello\nWorld\tTest",  # Newlines and tabs
        "Numbers: 123456789",
        "Unicode: 你好世界 🌍",
    ]

    print("\n2. Testing encode/decode round-trip:")
    print("-" * 80)

    for test_idx, text in enumerate(test_texts, 1):
        # Display text (truncate if too long)
        display_text = repr(text) if len(text) <= 40 else f"{repr(text[:40])}..."

        try:
            # Encode
            token_ids = bpe.encode(text)

            # Decode
            decoded_text = bpe.decode(token_ids)

            # Verify round-trip
            matches = decoded_text == text
            status = "✓" if matches else "✗"

            print(f"{test_idx:2d}. {status} Text: {display_text}")
            print(f"     → Tokens: {len(token_ids)} tokens")

            if not matches:
                print(f"     ✗ MISMATCH!")
                print(f"       Original: {repr(text)}")
                print(f"       Decoded:  {repr(decoded_text)}")

        except NotImplementedError:
            print(f"{test_idx:2d}. ⚠ Text: {display_text}")
            print(f"     → NotImplementedError: encode/decode not yet implemented")
            break
        except Exception as e:
            print(f"{test_idx:2d}. ✗ Text: {display_text}")
            print(f"     → Error: {type(e).__name__}: {e}")

    print("-" * 80)

    for text in test_texts:

        # Test token ID details for a simple example
        print(f"\n3. Detailed token breakdown for '{text}':")
        print("-" * 80)

        try:
            token_ids = bpe.encode(text)

            print(f"Text: {repr(text)}")
            print(f"Token IDs: {token_ids}")
            print(f"\nToken-by-token breakdown:")

            for idx, token_id in enumerate(token_ids):
                token_bytes = bpe.vocab.get(token_id, b'<UNKNOWN>')
                # Try to decode as UTF-8, fall back to hex representation
                try:
                    token_str = token_bytes.decode('utf-8')
                    token_repr = repr(token_str)
                except UnicodeDecodeError:
                    token_repr = f"<bytes: {token_bytes.hex()}>"

                print(f"  {idx}: ID={token_id:5d} → {token_repr:20s} ({token_bytes!r})")

            decoded = bpe.decode(token_ids)
            print(f"\nDecoded: {repr(decoded)}")
            print(f"Match: {decoded == text}")

        except NotImplementedError:
            print("⚠ encode/decode methods not yet implemented")
        except Exception as e:
            print(f"✗ Error: {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
