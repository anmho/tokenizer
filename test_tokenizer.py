from tokenizer import RegexTokenizer



def test_RegexTokenizer_merge():
    tokenizer = RegexTokenizer()

    merged = tokenizer._merge([10, 10, 20, 20, 30, 30], (10, 10), 256)

    assert merged == [256, 20, 20, 30, 30]

# def test_RegexTokenizer_encode_decode():
    # with open("tests/taylorswift.txt", "r") as f:
    #     train_text = f.read()

    # tokenizer = RegexTokenizer()
    # tokenizer.train(train_text, vocab_size=3000)

    # ids = tokenizer.encode("Taylor Swift")
    # tokens = tokenizer.translate(ids)
    # print(f"{tokens=}")
    # print(f"{tokenizer.encode('Taylor Swift')=}")
    # print(f"{tokenizer.decode(ids)=}")

    # tokenizer = RegexTokenizer()

    # with open("test/taylorswift", "r") as f:
    #     train_text = f.read()

    # tokenizer.train(train_text, vocab_size=3000)
    
