# Tokenizer Playground

Educational implementations of tokenization algorithms with a focus on
Byte Pair Encoding (BPE), trie-based greedy matching, and regex-based
pre-tokenization. This repo is intended for experimentation and learning.

## Contents

- `tokenizer.py` – core implementations and a small timing harness
- `tests/kingjames.txt` – sample training corpus
- `tests/prompt.txt` – small prompt for quick benchmarks
- `tests/taylorswift.txt` – additional sample text

## Tokenizers

The file `tokenizer.py` contains a few different encoder styles. This
section describes the core behavior and trade-offs of each:

- `SlowTokenizer`
  - Naive BPE merge replay over the full byte sequence.
  - Simple and easy to reason about, but slow for large inputs.
- `BasicTokenizer`
  - Span-based, rank-guided BPE (tiktoken-style idea).
  - Faster than merge replay while keeping BPE correctness.
- `TrieTokenizer`
  - Greedy longest-match over a trie built from the vocab.
  - Very fast, but not equivalent to rank-guided BPE (can diverge).
- `RegexTokenizer`
  - Regex pre-tokenization into segments, then BPE on each segment.
  - Trades extra regex overhead for smaller BPE chunks.

Note: greedy trie matching is not equivalent to rank-guided BPE and can
produce different tokenizations.

## Usage

Run the demo benchmark:

```
python tokenizer.py
```

This will:

- load/train tokenizers (using `*.merges.pkl` files)
- encode a sample prompt
- print timing and token counts

## Data Files

If you remove `basic_tokenizer.merges.pkl` or `regex_tokenizer.merges.pkl`,
the next run will retrain and recreate them.

## Notes

- This is a learning project, not a production tokenizer.
- Performance depends heavily on input size and Python overhead.
