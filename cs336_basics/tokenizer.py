import autoroot




class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        return 

    def encode(self, text:str) -> list[int]:

        return

    def encode_iterable(self, ):

        return

    def decode(self, ids: list[int]) -> str:
        return