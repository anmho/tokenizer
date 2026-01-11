from tokenizer import RegexTokenizer



def test_merge():
    tokenizer = RegexTokenizer()

    merged = tokenizer._merge([10, 10, 20, 20, 30, 30], (10, 10), 256)

    assert merged == [256, 20, 20, 30, 30]
