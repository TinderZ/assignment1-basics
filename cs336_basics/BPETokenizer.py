import torch
import regex as re
from pretokenization_example import find_chunk_boundaries

# GPT-2使用的预分词正则表达式模式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_BPE_tokenizer(input_path: str,vocab_size:int, special_tokens:list[str]):
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

    # 1.Vocabulary Initialization
    vocab = {i: bytes([i]) for i in range(256)} # inital vocab is the set of all bytes (utf-8)
    next_token_idx = 256 # next token index to assign
    merges = []


    # 2.Special Tokens  
    for special_token in special_tokens:
        vocab[next_token_idx] = special_token.encode("utf-8")
        next_token_idx += 1

    # 创建用于分割特殊标记的正则表达式
    # 使用 re.escape 确保特殊字符被正确转义，用 "|" 连接所有特殊标记
    if special_tokens:
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
    else:
        special_pattern = None
        
    # 统计每个pre-token（作为字节元组）的出现次数
    word_counts = {}
    
    with open(input_path, "rb") as file:
        num_processes = 4 #TODO
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")


        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")

            # 先按特殊标记分割chunk，避免跨特殊标记边界进行合并
            if special_pattern:
                text_segments = re.split(special_pattern, chunk)
            else:
                text_segments = [chunk]
            
            # 2.Pre-tokenization
            # 对每个分割后的文本段落进行预分词
            for segment in text_segments:
                if not segment:  # 跳过空字符串
                    continue
                    
                pre_tokens = re.finditer(PAT, segment)
                
                for match in pre_tokens:
                    pre_token_text = match.group()
                    # 将pre-token编码为UTF-8字节，并转为元组形式
                    token_bytes = pre_token_text.encode("utf-8")
                    # 将字节序列转为元组，每个元素是单字节的bytes对象
                    word_tuple = tuple(bytes([b]) for b in token_bytes)
                    
                    # 统计该word出现的次数
                    word_counts[word_tuple] = word_counts.get(word_tuple, 0) + 1

    # 3.Compute BPE Merges
    num_merges = vocab_size - len(vocab)  # 需要进行的合并次数
    
    # 初始化：统计所有相邻字节对的频率
    pair_counts = {}
    for word_tuple, count in word_counts.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    
    for _ in range(num_merges):
        # 如果没有字节对可以合并，退出
        if not pair_counts:
            break
        
        # 选择频率最高的字节对，如果有平局则选择字典序最大的
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        best_pair = best_pair[0]  # 获取字节对(tuple)
        
        # 将最优字节对加入merges列表
        merges.append(best_pair)
        
        # 将合并后的新token加入vocab
        vocab[next_token_idx] = best_pair[0] + best_pair[1]
        next_token_idx += 1
        
        # 更新word_counts和pair_counts，只处理包含best_pair的word
        new_word_counts = {}
        for word_tuple, count in word_counts.items():
            # 检查该word是否包含best_pair
            has_best_pair = False
            for i in range(len(word_tuple) - 1):
                if word_tuple[i] == best_pair[0] and word_tuple[i+1] == best_pair[1]:
                    has_best_pair = True
                    break
            
            if not has_best_pair:
                # 该word不包含best_pair，直接保留
                new_word_counts[word_tuple] = count
            else:
                # 该word包含best_pair，需要合并并更新pair_counts
                # 首先减少旧word的所有pair计数
                for i in range(len(word_tuple) - 1):
                    old_pair = (word_tuple[i], word_tuple[i+1])
                    pair_counts[old_pair] -= count
                    if pair_counts[old_pair] <= 0:
                        del pair_counts[old_pair]
                
                # 合并word_tuple中的best_pair
                new_word_tuple = []
                i = 0
                while i < len(word_tuple):
                    if i < len(word_tuple) - 1 and word_tuple[i] == best_pair[0] and word_tuple[i+1] == best_pair[1]:
                        new_word_tuple.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_word_tuple.append(word_tuple[i])
                        i += 1
                
                new_word_tuple = tuple(new_word_tuple)
                new_word_counts[new_word_tuple] = new_word_counts.get(new_word_tuple, 0) + count
                
                # 增加新word的所有pair计数
                for i in range(len(new_word_tuple) - 1):
                    new_pair = (new_word_tuple[i], new_word_tuple[i+1])
                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + count
        
        word_counts = new_word_counts

    return vocab, merges