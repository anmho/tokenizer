import time
from typing import List
from collections import Counter
import regex as re



class RegexTokenizer:
    def __init__(self):
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def _count_bigrams(self, tokens: List[int]) -> Counter:
        return Counter(list(zip(tokens, tokens[1:])))

    def _merge(self, tokens, pair, replacement) -> List[int]:
        i = 0

        result = []
        while i < len(tokens):
            # enumerate over bigrams and see if it matches the input pair
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                # replace the pair with replacement
                result.append(replacement)
                i += 2 # skip both tokens
            else:
                result.append(tokens[i])
                i += 1
        
        return result

    def _build_vocab(self, merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
        vocab = {token: bytes([token]) for token in range(256)} # vocab starts with original tokens, trains a new vocab from the merges. these new tokens are LEARNED
        # sorted(merges.items(), key=lambda x: x[1]) # sort by replacement index so we have it in order. technically not necessary
        for pair, token in merges.items():
            p0, p1 = pair
            vocab[token] = vocab[p0] + vocab[p1] # this is kind of dynamic programming

        return vocab

    def _segment(self, text: str) -> List[str]:
        return re.findall(self.pat, text)
    def _token_ids(self, text: str) -> List[int]:
        ids = list(map(int, text.encode("utf-8")))
        return ids

    def train(self, text: str, vocab_size: int, verbose=False):
        """ trains the vocab map after merging and the merge sequence"""
        # steps

        # convert from text to raw tokens
        # e.g. utf-8 bytes

        segments = self._segment(text)
        segments_ids = [self._token_ids(segment) for segment in segments]

        original_token_count = sum(len(tokens) for tokens in segments_ids)

        merges = {}
        vocab = {token: bytes([token]) for token in range(256)} # vocab starts 
        # how many merges do we need?

        num_merges = vocab_size - 256
        print(f"{num_merges=}")

        # bigram counts across segments
        global_counter = Counter()
        for ids in segments_ids:
            bigram_counts = self._count_bigrams(ids)
            global_counter += bigram_counts

        for i in range(num_merges):
            if len(global_counter) == 0:
                break
            pair, freq = global_counter.most_common(n=1)[0]
            replacement = i + 256
            print(f"merge {i}: merging {pair} with {replacement}: {freq=}")
            # update merge map
            merges[pair] = replacement


            # updated_tokens = []
            for i, ids in enumerate(segments_ids):
                merged = self._merge(ids, pair, replacement)
                if len(ids) != len(merged):
                    segments_ids[i] = merged
                    old_counts = self._count_bigrams(ids) # remove old counts
                    new_counts = self._count_bigrams(merged) # remove old counts
                    global_counter -= old_counts
                    global_counter += new_counts

                    vocab[replacement] = vocab[pair[0]] + vocab[pair[1]]
                    print(f"create new vocab {vocab[pair[0]] + vocab[pair[1]]}")

        vocab = self._build_vocab(merges)
        self.merges = merges
        self.vocab = vocab
        merged_token_count = sum(len(tokens) for tokens in segments_ids)
        print(f"vocab size: {len(vocab)}")
        print(f"{len(merges)=}")
        print(f"original token count: {original_token_count}")
        print(f"merged token count: {merged_token_count}")
        print(f"compression ratio: {original_token_count / merged_token_count}")


    def encode(self, text: str) -> List[int]:
        segments = self._segment(text)
        all_ids = []
        for segment in segments:
            ids = list(map(int, segment.encode("utf-8")))

            for pair, id in self.merges.items():
                ids = self._merge(ids, pair, id)

            all_ids.extend(ids)

        return all_ids
    
    def translate(self, ids) -> List[str]:
        tokens = []
        for id in ids:
            token = self.vocab[id]
            tokens.append(token.decode("utf-8"))
        
        return tokens

    def decode(self, ids: List[int]) -> str:
        res = []
        for id in ids:
            # get the byte mapping for the id from raw utf8 bytes
            res.append(self.vocab[id])

        return b"".join(res).decode("utf-8", errors="replace") # sometimes llm can output an invalid utf-8 sequence

if __name__ == "__main__":
    with open("tests/taylorswift.txt", "r") as f:
        train_text = f.read()

    tokenizer = RegexTokenizer()
    tokenizer.train(train_text, vocab_size=3000)

    ids = tokenizer.encode("Taylor Swift")
    tokens = tokenizer.translate(ids)
    print(f"{tokens=}")
    print(f"{tokenizer.encode('Taylor Swift')=}")
    print(f"{tokenizer.decode(ids)=}")