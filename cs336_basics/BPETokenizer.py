import autoroot
import torch
import regex as re
from pretokenization_example import find_chunk_boundaries

# GPT-2使用的预分词正则表达式模式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _initialize_vocab_with_special_tokens(special_tokens: list[str]) -> tuple[dict, int]:
    """初始化词汇表并添加特殊标记"""
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_idx = 256
    
    for special_token in special_tokens:
        vocab[next_token_idx] = special_token.encode("utf-8")
        next_token_idx += 1
    
    return vocab, next_token_idx


def _build_special_pattern(special_tokens: list[str]) -> str | None:
    """创建用于分割特殊标记的正则表达式"""
    if special_tokens:
        return "|".join(re.escape(token) for token in special_tokens)
    return None


def _count_word_frequencies(input_path: str, special_pattern: str | None) -> dict:
    """统计每个pre-token（作为字节元组）的出现次数"""
    word_counts = {}
    
    with open(input_path, "rb") as file:
        num_processes = 4
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")

            # 先按特殊标记分割chunk，避免跨特殊标记边界进行合并
            if special_pattern:
                text_segments = re.split(special_pattern, chunk)
            else:
                text_segments = [chunk]
            
            # 对每个分割后的文本段落进行预分词
            for segment in text_segments:
                if not segment:
                    continue
                    
                pre_tokens = re.finditer(PAT, segment)
                
                for match in pre_tokens:
                    pre_token_text = match.group()
                    token_bytes = pre_token_text.encode("utf-8")
                    word_tuple = tuple(bytes([b]) for b in token_bytes)
                    word_counts[word_tuple] = word_counts.get(word_tuple, 0) + 1
    
    return word_counts


def _compute_initial_pair_counts(word_counts: dict) -> dict:
    """初始化：统计所有相邻字节对的频率"""
    pair_counts = {}
    for word_tuple, count in word_counts.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts


def _merge_pair_in_word(word_tuple: tuple, best_pair: tuple) -> tuple:
    """在word_tuple中合并所有的best_pair"""
    new_word_tuple = []
    i = 0
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and word_tuple[i] == best_pair[0] and word_tuple[i+1] == best_pair[1]:
            new_word_tuple.append(best_pair[0] + best_pair[1])
            i += 2
        else:
            new_word_tuple.append(word_tuple[i])
            i += 1
    return tuple(new_word_tuple)


def _update_counts_after_merge(word_counts: dict, pair_counts: dict, best_pair: tuple) -> dict:
    """在一次合并后更新word_counts和pair_counts"""
    new_word_counts = {}
    
    for word_tuple, count in word_counts.items():
        # 检查该word是否包含best_pair
        has_best_pair = any(
            word_tuple[i] == best_pair[0] and word_tuple[i+1] == best_pair[1]
            for i in range(len(word_tuple) - 1)
        )
        
        if not has_best_pair:
            new_word_counts[word_tuple] = count
        else:
            # 减少旧word的所有pair计数
            for i in range(len(word_tuple) - 1):
                old_pair = (word_tuple[i], word_tuple[i+1])
                pair_counts[old_pair] -= count
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
            
            # 合并word_tuple中的best_pair
            new_word_tuple = _merge_pair_in_word(word_tuple, best_pair)
            new_word_counts[new_word_tuple] = new_word_counts.get(new_word_tuple, 0) + count
            
            # 增加新word的所有pair计数
            for i in range(len(new_word_tuple) - 1):
                new_pair = (new_word_tuple[i], new_word_tuple[i+1])
                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + count
    
    return new_word_counts


def _perform_bpe_merges(word_counts: dict, num_merges: int, vocab: dict, next_token_idx: int) -> tuple[list, int]:
    """执行BPE合并过程"""
    merges = []
    pair_counts = _compute_initial_pair_counts(word_counts)
    
    for _ in range(num_merges):
        if not pair_counts:
            break
        
        # 选择频率最高的字节对，如果有平局则选择字典序最大的
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        
        merges.append(best_pair)
        vocab[next_token_idx] = best_pair[0] + best_pair[1]
        next_token_idx += 1
        
        word_counts = _update_counts_after_merge(word_counts, pair_counts, best_pair)
    
    return merges, next_token_idx


def train_BPE_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    Train a Byte-Pair Encoding (BPE) Tokenizer on the input text.

    Args:
        input_path: The path to the input text file.
        vocab_size: The total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: A list of strings to be added to the tokenizer vocabulary. These special tokens do not otherwise affect BPE training. 

    Returns:
        vocab: The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges: A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
    """
    # 1. 初始化词汇表并添加特殊标记
    vocab, next_token_idx = _initialize_vocab_with_special_tokens(special_tokens)
    
    # 2. 创建特殊标记的正则表达式模式
    special_pattern = _build_special_pattern(special_tokens)
    
    # 3. 统计词频 Pre-tokenization
    word_counts = _count_word_frequencies(input_path, special_pattern)
    
    # 4. 执行BPE merge
    num_merges = vocab_size - len(vocab)
    merges, next_token_idx = _perform_bpe_merges(word_counts, num_merges, vocab, next_token_idx)
    
    return vocab, merges