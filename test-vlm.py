import sys
import torch
from PIL import Image

from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig
)

MODEL_ID = "openbmb/MiniCPM-V-2_6"

print("[Info] モデルロード中...")

# ==========================================
# 4bit量子化
# ==========================================
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ==========================================
# tokenizer
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

# ==========================================
# model
# ==========================================
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa"
).eval()

print("[Info] ロード完了")


def run_ocr(image_path):

    image = Image.open(image_path).convert("RGB")

    msgs = [
        {
            "role": "user",
            "content": [
                image,
                (
                    "この画像は英単語帳です。"
                    "番号、英単語、日本語をCSV形式で抽出してください。\n\n"
                    "番号,英単語,日本語"
                )
            ]
        }
    ]

    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )

    return res


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage: python test-vlm.py image.png")
        sys.exit(1)

    img_path = sys.argv[1]

    result = run_ocr(img_path)

    print(result)

    with open(
        "last_result.csv",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(result)