import os

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"



from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B"
)


def count_tokens(text):

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    return len(tokens)
