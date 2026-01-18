from dataclasses import dataclass
import pickle
import time
from typing import List
from collections import Counter
from typing_extensions import override
import regex as re

def span_encode(text: str, lookup_token: dict):
    text_bytes = text.encode("utf-8")

    # use spans which are initialized as pairs
    spans = [(i, i+1) for i in range(len(text_bytes))]


    def pair_rank(i):
        if i < 0 or i >= len(spans) - 1:
            return float("inf")
        s0, e0 = spans[i] # span 0 (left)
        s1, e1 = spans[i+1] # span 1 (right)

        # get the rank (token_id) of the token by direct lookup of the bytes
        # pair = (text_bytes[s0:e0], text_bytes[s1:e1])

        merged_token = text_bytes[s0:e1]

        rank = lookup_token.get(merged_token, float("inf"))
        return rank

    # len(span) -1 because len(pairs) == len(span) - 1
    # len(span) == len(token_ids) (initially)
    ranks = [pair_rank(i) for i in range(len(spans)-1)] # calculate the rank for every possible pair of spans
    while ranks:
        # find the lowest pair
        # best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
        i = min(range(len(ranks)), key=ranks.__getitem__) # i = argmin(ranks)
        if ranks[i] == float("inf"):
            # there are no mergeable ranks in the spans we have
            break

        # token_ids = self.merge(token_ids, best_pair, rank)
        # merge i and i +1 spans
        spans[i] = (spans[i][0], spans[i+1][1])
        spans.pop(i+1)

        # remove rank for merged pair
        ranks.pop(i)

        # update neighbor ranks
        if i - 1 >= 0:
            ranks[i-1] = pair_rank(i-1)
        if i < len(spans) - 1: # i is the index of the first element of the new span was just created after merging
            ranks[i] = pair_rank(i)
            
    # for pair, id in self.merges.items():
        # token_ids = self.merge(token_ids, pair, id)

    return [lookup_token[text_bytes[s:e]] for s, e in spans]

class SlowTokenizer:
    def count_bigrams(self, token_ids: List[int]) -> Counter:
        return Counter(list(zip(token_ids, token_ids[1:])))

    # naive merge. find all instances of pair in tokens and replace it.
    def merge(self, tokens, pair, replacement) -> List[int]:
        i = 0

        result = []
        while i < len(tokens):
            # enumerate over bigrams and see if it matches the input pair
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                result.append(replacement)
                i += 2 # skip both tokens
            else:
                result.append(tokens[i])
                i += 1
        
        return result

    def build_vocab(self, merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
        vocab = {token: bytes([token]) for token in range(256)} # vocab starts with original tokens, trains a new vocab from the merges. these new tokens are LEARNED
        # sorted(merges.items(), key=lambda x: x[1]) # sort by replacement index so we have it in order. technically not necessary
        for pair, token in merges.items():
            p0, p1 = pair
            vocab[token] = vocab[p0] + vocab[p1] # this is kind of dynamic programming

        return vocab

    def text_to_token_ids(self, text: str) -> List[int]:
        ids = list(map(int, text.encode("utf-8")))
        return ids

    def load(self, merges_file="basic_tokenizer.merges.pkl"):
        with open(merges_file, "rb") as f:
            self.merges = pickle.load(f)

        self.vocab = self.build_vocab(self.merges)
        self.lookup_token = {token: token_id for token_id, token in self.vocab.items()}


    def save(self, merges_file="basic_tokenizer.merges.pkl"):
        with open(merges_file, "wb") as f:
            print(self.merges)
            pickle.dump(self.merges, f)

    def train(self, text: str, vocab_size: int, verbose: bool=False):
        """ trains the vocab map after merging and the merge sequence"""

        # steps

        # convert from text to raw tokens
        # e.g. utf-8 bytes

        token_ids = self.text_to_token_ids(text)
        original_token_count = len(token_ids)
        merges = {}
        vocab = {token: bytes([token]) for token in range(256)} # vocab starts 
        # how many merges do we need?

        num_merges = vocab_size - 256
        print(f"{num_merges=}")


        
        # bigram counts across segments
        for i in range(num_merges):
            bigram_counts = self.count_bigrams(token_ids)
            if len(bigram_counts) == 0:
                break

            pair, count = bigram_counts.most_common(n=1)[0]
            replacement = i + 256
            merges[pair] = replacement
            new_token_ids = self.merge(token_ids, pair, replacement)
            p0, p1 = pair
            vocab[replacement] = vocab[p0] + vocab[p1]

            print(f"create new vocab {vocab[p0] + vocab[p1]}")
            print(f"    new token count {len(new_token_ids)}")
            print(f"    new vocab size {len(vocab)}")
            
            token_ids = new_token_ids


        vocab = self.build_vocab(merges)
        self.merges = merges
        self.vocab = vocab
        self.lookup_token = {token: token_id for token_id, token in self.vocab.items()}

        new_token_count = len(token_ids)

        if verbose:
            print("Completed training:")
            print(f"    Original Token Count: {original_token_count}")
            print(f"    New Token Count: {new_token_count}")
            print(f"    Vocab Size: {len(vocab)}")

    def encode(self, text: str):
        ids = list(map( int, text.encode("utf-8")))

        while True:
            pairs = zip(ids, ids[1:])

            best_rank = float('inf')
            # find the best rank pair. we will merge that one
            best_pair = None

            for pair in pairs:
                rank = self.merges.get(pair)
                if rank is None:
                    continue

                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            # if best pair is none, there are no more mergeable pairs. lets go to the next segment
            if not best_pair:
                break
            new_ids = self.merge(ids, best_pair, best_rank)
            ids = new_ids
        return ids
    
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

class BasicTokenizer(SlowTokenizer):
    def encode(self, text: str) -> List[int]:
        return span_encode(text, self.lookup_token)
        
    
# @dataclass
# class Node:
#     token_id: int = 0
#     next: "Node" = None
#     prev: "Node" = None
# class HeapTokenizer(SlowTokenizer):
#     def build_pair_list(self, token_ids):
#         nodes = [Node(token_id) for token_id in token_ids]

#         for i in range(1, len(nodes)):
#             left, right = nodes[i-1], nodes[i]
#             left.next = right
#             right.prev = left

#         headDum = Node()
#         tailDum = Node()
#         headDum.next = tailDum
#         tailDum.prev = headDum

#         headDum.next = nodes[0]

        
            
#     def encode(self, text: str) -> List[int]:
#         token_ids = self.text_to_token_ids(text)
#         pairs = list(zip(token_ids, token_ids[1:]))

#         headDum, tailDum, nodes = self.build_pair_list(pairs)

#         minHeap = []
#         for pair in pairs:
#             if pair in self.merges:
#                 token_id = self.merges[pair] # e.g. rank
#                 minHeap.append((token_id, pair))
        
#         def merge(left, right):
#             token_id = self.merges[token_id]


#             # left of pair
#             pre = left.prev
#             # right of pair
#             post = right.next

            
#             merged = Node(token_id)
#             # connect before left to merged
#             pre.next = merged
#             merged.prev = pre

#             post.prev = merged
#             merged.next = post

#             token_id = self.merges[pair]

#             return merged


#         heapq.heapify(minHeap)
#         while minHeap:
#             token_id, pair = heapq.heappop(minHeap)
#             # replace pair with token_id
#             # this is selected to merge
#             # we need to replace pair with token_id

#             merged = merge(pair)


#             # need to add new pairs

#             prev = merged.prev
#             nxt = merged.next
#             if prev is not headDum:
#                 new_pair = (prev.token_id, merged.token_id)
#                 # check if its mergable
#                 if new_pair in self.merges:
#                     token_id = self.merges[new_pair]
#                     heapq.heappush(minHeap, (token_id, new_pair))

#             if nxt is not tailDum:
#                 new_pair = (merged.token_id, nxt.token_id)
#                 # check if its mergable
#                 if new_pair in self.merges:
#                     token_id = self.merges[new_pair]
#                     heapq.heappush(minHeap, (token_id, new_pair))

#         # collect all the values

#         cur = headDum.next
#         new_token_ids = []
#         while cur:
#             new_token_ids.append(cur.token_id)
#             cur = cur.next
#         return token_ids
        

@dataclass
class TrieNode:
    is_terminal: bool = False
    token_id: int = 0
    def __post_init__(self):
        self.children = {}

def build_trie(vocab):
    root = TrieNode()
    def insert(root: TrieNode, token_id: int, token: bytes):
        cur = root

        for b in token:
            if b not in cur.children:
                cur.children[b] = TrieNode()

            cur = cur.children[b]
        
        cur.is_terminal = True
        cur.token_id = token_id
        return root
            
    for token_id, token in vocab.items():
        insert(root, token_id, token)

    return root

def greedy_trie_match(i: int, trie: TrieNode, text_bytes: List[int]):
    j = i
    cur = trie
    last_match = None
    last_match_idx = None

    while cur and j < len(text_bytes):
        b = text_bytes[j]
        if b not in cur.children:
            # we cannot proceed
            break
        cur = cur.children[b]
        
        if cur.is_terminal:
            last_match = cur.token_id
            last_match_idx = j

        j += 1
    return last_match, last_match_idx


class TrieTokenizer(SlowTokenizer):
    
    @override
    def load(self):
        super().load()

        self.trie = build_trie(self.vocab)
    
    def encode(self, text: str):
        text_bytes = self.text_to_token_ids(text)
        # use a trie match

        i = 0
        encoded = []
        while i < len(text_bytes):
            trie = self.trie
            last_match, last_match_idx = greedy_trie_match(i, trie, text_bytes)
            if last_match is not None:
                encoded.append(last_match)
                i = last_match_idx + 1
            else:
                encoded.append(text_bytes[i])
                i += 1
        return encoded
    
        
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
                    new_counts = self._count_bigrams(merged) # add new counts
                    global_counter -= old_counts
                    global_counter += new_counts

                    vocab[replacement] = vocab[pair[0]] + vocab[pair[1]]
                    # print(f"create new vocab {vocab[pair[0]] + vocab[pair[1]]}")

        vocab = self._build_vocab(merges)
        self.merges = merges
        self.vocab = vocab
        merged_token_count = sum(len(tokens) for tokens in segments_ids)
        
        print(f"vocab size: {len(vocab)}")
        print(f"{len(merges)=}")
        print(f"original token count: {original_token_count}")
        print(f"merged token count: {merged_token_count}")
        print(f"compression ratio: {original_token_count / merged_token_count}")

        self.trie = build_trie(self.vocab)

        self.token_lookup = {token: token_id for token_id, token in self.vocab.items()}


    def load(self, merges_file="regex_tokenizer.merges.pkl"):
        with open(merges_file, "rb") as f:
            self.merges = pickle.load(f)

        self.vocab = self._build_vocab(self.merges)

        self.trie = build_trie(self.vocab)
        self.token_lookup = {token: token_id for token_id, token in self.vocab.items()}


        

    def save(self, merges_file="regex_tokenizer.merges.pkl"):
        with open(merges_file, "wb") as f:
            pickle.dump(self.merges, f)

    def encode(self, text: str) -> List[int]:
        segments = self._segment(text)
        all_ids = []
        for segment in segments:
            ids = span_encode(segment, self.token_lookup)
            all_ids.extend(ids)

            # ids = segment.encode("utf-8")
            # if ids in self.token_lookup: # full match
            #     # print(f"full match for {ids}")
            #     all_ids.append(self.token_lookup[ids])
            #     continue

            # while True:
            #     pairs = zip(ids, ids[1:])
            #     # find the best rank pair. we will merge that one
            #     best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            #     rank = self.merges.get(best_pair)
            #     if not rank: # no mergable ranks
            #         # if rank is none, there are no more mergeable pairs. lets go to the next segment
            #         break

            #     new_ids = self._merge(ids, best_pair, rank)
            #     ids = new_ids


            # all_ids.extend(ids)

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


def get_slow_tokenizer(train_text, vocab_size=15_000):
    basic_tokenizer = SlowTokenizer()
    try:
        basic_tokenizer.load()
    except:
        basic_tokenizer.train(train_text, vocab_size)
        basic_tokenizer.save()

    return basic_tokenizer

def get_basic_tokenizer(train_text, vocab_size=15_000):
    basic_tokenizer = BasicTokenizer()
    try:
        basic_tokenizer.load()
    except:
        basic_tokenizer.train(train_text, vocab_size)
        basic_tokenizer.save()

    return basic_tokenizer

# def get_heap_tokenizer(train_text, vocab_size=15_000):
#     heap_tokenizer = HeapTokenizer()
#     try:
#         heap_tokenizer.load()
#     except:
#         heap_tokenizer.train(train_text, vocab_size)
#         heap_tokenizer.save()

#     return heap_tokenizer

def get_trie_tokenizer(train_text, vocab_size=15_000):
    trie_tokenizer = TrieTokenizer()
    try:
        trie_tokenizer.load()
    except:
        trie_tokenizer.train(train_text, vocab_size)
        trie_tokenizer.save()

    return trie_tokenizer

def get_trie_tokenizer(train_text, vocab_size=15_000):
    trie_tokenizer = TrieTokenizer()
    try:
        trie_tokenizer.load()
    except:
        trie_tokenizer.train(train_text, vocab_size)
        trie_tokenizer.save()

    return trie_tokenizer

def get_regex_tokenizer(train_text, vocab_size=15_000):
    regex_tokenizer = RegexTokenizer()
    try:
        regex_tokenizer.load()
    except:
        regex_tokenizer.train(train_text, vocab_size)
        regex_tokenizer.save()

    return regex_tokenizer

def timed(f):
    start = time.time()
    v = f()
    elapsed = time.time() - start
    return v, elapsed

if __name__ == "__main__":
    with open("tests/kingjames.txt", "r") as f:
        train_text = f.read()
    
    with open("tests/prompt.txt", "r") as f:
        test_text = f.read()
    slow_tokenizer = get_slow_tokenizer(train_text)
    basic_tokenizer = get_basic_tokenizer(train_text)
    
    trie_tokenizer = get_trie_tokenizer(train_text)
    regex_tokenizer = get_regex_tokenizer(train_text)

    # heap_tokenizer = get_heap_tokenizer(train_text)

    with open("tests/taylorswift.txt", "r") as f:
        prompt = f.read()


    print("starting")
    tokens_1, elapsed = timed(lambda: slow_tokenizer.encode(test_text))
    print(f"SlowTokenizer took {elapsed} seconds for tokenizer.encode()")
    tokens_2, elapsed = timed(lambda: basic_tokenizer.encode(test_text))
    print(f"BasicTokenizer took {elapsed} seconds for tokenizer.encode()")
    
    tokens_3, elapsed = timed(lambda: trie_tokenizer.encode(test_text))
    print(f"TrieTokenizer took {elapsed} seconds for tokenizer.encode()")
    tokens_4, elapsed = timed(lambda: regex_tokenizer.encode(test_text))
    print(f"RegexTokenizer took {elapsed} seconds for tokenizer.encode()")

    # tokens_5, elapsed = timed(lambda: heap_tokenizer.encode(test_text))
    # print(f"HeapTokenizer took {elapsed} seconds for tokenizer.encode()")

    print("SlowTokenizer results: ", len(tokens_1))
    print("BasicTokenizer results: ", len(tokens_2))
    
    print("TrieTokenizer results: ", len(tokens_3))
    print("RegexTokenizer results: ", len(tokens_4))

    # print("HeapTokenizer results: ", tokens_5)

    # assert tokens_1 == tokens_2 == tokens_3
    
    