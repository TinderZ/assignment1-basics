import pickle

# 读取 vocab
path = "res/TS/vocab.pkl"

with open(path, "rb") as f:
    vocab = pickle.load(f)

# 查看前20个条目
print(f"词汇表大小: {len(vocab)}")

print("\n前20个词汇:")
for i, (token, idx) in enumerate(list(vocab.items())[20:]):
    print(f"{idx}: {repr(token)}")


# 查找最长的标记
print("\n" + "=" * 80)
print("查找最长的标记...")
longest_token = max(vocab.values(), key=len)
print(f"最长的token: {repr(longest_token)}")
print(f"长度: {len(longest_token)}")
