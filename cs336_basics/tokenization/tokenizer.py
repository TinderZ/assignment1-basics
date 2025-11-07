import autoroot
from numpy import byte
from cs336_basics.BPETokenizer import PAT
import pickle
import regex as re
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens: list[str]|None) -> None:
        self.id_to_bytes = vocab
        self.vocab_len = len(vocab)
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.pair_ranks = {pair: i for i, pair in enumerate(merges)}

        # 记录特殊标记，并编译“保留分隔符”的正则（使用捕获分组）
        self.special_tokens = special_tokens or []
        self.special_tokens_re = None
        if self.special_tokens:
            # 为避免前缀遮挡，按长度从长到短排序
            _sorted = sorted(self.special_tokens, key=len, reverse=True)
            _pattern = "(" + "|".join(re.escape(tok) for tok in _sorted) + ")"
            self.special_tokens_re = re.compile(_pattern)

            # 确保所有特殊标记的 bytes 都在词表中
            for _token in self.special_tokens:
                _bytes = _token.encode("utf-8")
                if _bytes not in self.bytes_to_id:
                    self.id_to_bytes[self.vocab_len] = _bytes
                    self.bytes_to_id[_bytes] = self.vocab_len
                    self.vocab_len += 1


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]=None): 
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath,"rb") as f:
            merges = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens) 


    def merge_bytes_to_token_BPE(self, init_bytes: bytes) -> list[bytes]:
        """
            args:
                init_bytes: 传入的分割后的bytes

            returns:
                bytes_to_token
        """
        # Fast path: 整个bytes就是一个已存在的token 成本低、收益明显
        if init_bytes in self.bytes_to_id:
            return [init_bytes]

        bytes_to_token = [bytes([b]) for b in init_bytes] 
        if len(bytes_to_token) <= 1: return bytes_to_token

        while True:
            best_pair = None
            best_rank = None
            # 选最优相邻对
            for i in range(len(bytes_to_token) - 1):
                pair = (bytes_to_token[i], bytes_to_token[i + 1])
                rank = self.pair_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair, best_rank = pair, rank

            if best_pair is None:
                break

            # 合并该对的所有不重叠出现
            new_bytes_to_token: list[bytes] = []
            i = 0
            while i < len(bytes_to_token):
                if i < len(bytes_to_token) - 1 and bytes_to_token[i] == best_pair[0] and bytes_to_token[i + 1] == best_pair[1]:
                    new_bytes_to_token.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_bytes_to_token.append(bytes_to_token[i])
                    i += 1
            bytes_to_token = new_bytes_to_token


            if len(bytes_to_token) == 1:
                break

        return bytes_to_token


    def encode(self, text: str) -> list[int]:
        idxs: list[int] = []
        # 先用“带捕获分组”的特殊标记正则切分文本，并保留这些标记
        text_segments = (
            self.special_tokens_re.split(text) if self.special_tokens_re is not None else [text]
        )

        # 对每个片段分别处理：特殊标记直接映射；普通片段走预分词+合并
        for segment in text_segments:
            if not segment:
                continue
            if segment in self.special_tokens:
                idxs.append(self.bytes_to_id[segment.encode("utf-8")])
                continue
            for m in re.finditer(PAT, segment):
                bt = m.group().encode("utf-8")
                for tok in self.merge_bytes_to_token_BPE(bt):
                    idxs.append(self.bytes_to_id[tok])
        return idxs



    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for tid in self.encode(chunk):
                yield tid


    def decode(self, ids: list[int]) -> str:
        bt = b"".join(self.id_to_bytes.get(tid, b"") for tid in ids)
        return bt.decode("utf-8", errors="replace")

if __name__ == "__main__":
    tok = Tokenizer.from_files("res/TS/vocab.pkl", "res/TS/merges.pkl", special_tokens=["<|endoftext|>"])
    print(tok.encode(" , the cat ate"))
    print(tok.decode(tok.encode(" , the cat ate")))