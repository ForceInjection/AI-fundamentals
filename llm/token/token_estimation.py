import tiktoken
from transformers import BertTokenizer
import argparse

def estimate_tokens_tiktoken(text, model="cl100k_base"):  # 修改模型名称
    """
    使用 tiktoken 估算 Tokens 数量
    """
    try:
        encoder = tiktoken.get_encoding(model)
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error in tiktoken estimation: {e}")
        print(f"tiktoken version: {tiktoken.__version__}")
        return None

def estimate_tokens_transformers(text):
    """
    使用 transformers 估算中文 Tokens 数量
    """
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        tokens = tokenizer.encode(text, add_special_tokens=False)  # 去掉特殊字符
        return len(tokens)
    except Exception as e:
        print(f"Error in transformers estimation: {e}")
        return None

def main():
    english_text = "What is the capital of China?"
    chinese_text = "《哪吒之魔童闹海》于2025年1月29日上映，讲述了哪吒在成长过程中，面对重重困难，不断挑战自我，最终逆天改命的故事。影片融合了精彩的特效和深刻的情感，展现了哪吒的勇敢与坚韧。"

    print("\n--- 英文 Tokens 估算 ---")
    print(f"输入文本: {english_text}")
    tiktoken_eng = estimate_tokens_tiktoken(english_text)
    print(f"Tokens 数量（tiktoken）: {tiktoken_eng}")

    print("\n--- 中文 Tokens 估算 ---")
    print(f"输入文本: {chinese_text}")
    transformers_chinese = estimate_tokens_transformers(chinese_text)
    print(f"Tokens 数量（transformers）: {transformers_chinese}")
    tiktoken_chinese = estimate_tokens_tiktoken(chinese_text)
    print(f"Tokens 数量（tiktoken）: {tiktoken_chinese}")

if __name__ == "__main__":
    main()