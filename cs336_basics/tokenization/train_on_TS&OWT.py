import autoroot
from cs336_basics.BPETokenizer import train_BPE_tokenizer
import cProfile
import pstats
import pickle
import json


def main():
    # 使用验证集作为debug dataset
    input_path = "data/TinyStoriesV2-GPT4-train.txt"  #data\owt_train.txt
    vocab_size = 10000  # 32000
    special_tokens = ["<|endoftext|>"]   #"\n"
    
    print(f"开始在验证集上训练 BPE tokenizer...")
    print(f"输入文件: {input_path}")
    print(f"词汇表大小: {vocab_size}")
    print(f"特殊标记: {special_tokens}")
    
    # 使用 cProfile 进行性能分析
    profiler = cProfile.Profile()
    profiler.enable()
    
    vocab, merges = train_BPE_tokenizer(input_path, vocab_size, special_tokens)
    
    profiler.disable()
    
    print(f"\n训练完成!")
    print(f"最终词汇表大小: {len(vocab)}")
    print(f"合并次数: {len(merges)}")
    
    # 序列化结果
    with open("res/TS/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    with open("res/res-TS/merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    
    print(f"\n已保存 vocab.pkl 和 merges.pkl")
    
    # 输出性能分析结果
    print("\n" + "="*80)
    print("性能分析结果 - 按累计时间排序 (前30个函数):")
    print("="*80)
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    
    print("\n" + "="*80)
    print("性能分析结果 - 按总时间排序 (前30个函数):")
    print("="*80)
    stats.sort_stats('tottime')
    stats.print_stats(30)
    
    # 保存性能分析结果到文件
    with open("res/TS/profile_stats.txt", "w", encoding="utf-8") as f:
        stats = pstats.Stats(profiler, stream=f)
        f.write("="*80 + "\n")
        f.write("按累计时间排序:\n")
        f.write("="*80 + "\n")
        stats.sort_stats('cumulative')
        stats.print_stats()
        f.write("\n" + "="*80 + "\n")
        f.write("按总时间排序:\n")
        f.write("="*80 + "\n")
        stats.sort_stats('tottime')
        stats.print_stats()
    
    print(f"\n完整的性能分析结果已保存到 profile_stats.txt")


if __name__ == "__main__":
    main()
