
from cs336_basics.assignment_1.bpe_tokenizer import BPETokenizer
from cs336_basics.constants import GPT2_PAT_STR, MODEL_DEFAULTS, TINYSTORIES_VALID, OWT_TRAIN


if __name__ == "__main__":
    bpe = BPETokenizer(GPT2_PAT_STR, MODEL_DEFAULTS.DEFAULT_SPECIAL_TOKENS, verbose=True)
    train_result = bpe.train(OWT_TRAIN, 32_000)
    print('vocab size = ', len(train_result.vocab))
    print('merges count = ', len(train_result.merges))
    print('First 30 longest merges:', sorted(train_result.merges, key=lambda x: len(x[0] + x[1]), reverse=True)[:30])