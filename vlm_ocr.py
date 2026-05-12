import csv
import io
import json
import re
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_ID = "openbmb/MiniCPM-V-2_6"
DEFAULT_PROMPT_KEY = "blue_word_cards"
DEFAULT_PROMPT = (
    "この画像は日本の英単語帳のページです。"
    "抽出対象は、単語カード形式の見出し語だけです。"
    "各カードは、左端の縦長の番号帯、中央の大きな英単語、右側の日本語意味というレイアウトを持ちます。"
    "カラー画像では青い見出し枠に見えますが、白黒PDFでは色ではなくこのレイアウトで判定してください。"
    "各カードから次の3項目だけ抽出してください: "
    "number=番号帯の単語番号, english=太字の英単語, japanese=日本語意味。"
    "日本語意味が複数あるときは必ず「、」区切りで1つの文字列にまとめてください。"
    "日本語意味の中に {for} や 《to》 のような英語の補足があれば、その括弧内は削除してください。"
    "品詞記号、発音記号、例文、派生語、関連語、熟語、QRコード、ページ番号は不要です。"
    "長文ページ、会話文、subコラム、Check!!、チェックリスト、補足欄しか写っていない場合は空配列を返してください。"
    "出力はJSON配列のみ。各要素は "
    '{"number":"312","english":"especially","japanese":"特に"} '
    "の形にしてください。説明文やMarkdownコードフェンスは不要です。"
)


def load_prompt(prompt_path: str, prompt_key: str) -> str:
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_PROMPT

    prompt_block = payload.get(prompt_key)
    if isinstance(prompt_block, dict):
        prompt = prompt_block.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()

    if isinstance(prompt_block, str) and prompt_block.strip():
        return prompt_block.strip()

    return DEFAULT_PROMPT


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def remove_english_annotations(text: str) -> str:
    bracket_patterns = [
        (r"\{([^{}]*)\}", "{}"),
        (r"｛([^｛｝]*)｝", "｛｝"),
        (r"《([^《》]*)》", "《》"),
        (r"〈([^〈〉]*)〉", "〈〉"),
        (r"<([^<>]*)>", "<>"),
    ]
    english_note_pattern = re.compile(r"[A-Za-z0-9\s.,;:/&+!?'\"~_-]+")

    cleaned = text
    for pattern, _ in bracket_patterns:
        def replacer(match: re.Match[str]) -> str:
            inner = match.group(1).strip()
            if inner and english_note_pattern.fullmatch(inner):
                return ""
            return match.group(0)

        cleaned = re.sub(pattern, replacer, cleaned)

    return cleaned


def normalize_japanese_meaning(text: str) -> str:
    cleaned = remove_english_annotations(text)
    cleaned = re.sub(r"\s*[;；,，/／]\s*", "、", cleaned)
    cleaned = re.sub(r"\s*、\s*", "、", cleaned)
    parts = [part.strip() for part in cleaned.split("、")]

    normalized_parts: List[str] = []
    seen = set()
    for part in parts:
        part = part.strip(" 　")
        if not part:
            continue
        if part not in seen:
            normalized_parts.append(part)
            seen.add(part)

    return "、".join(normalized_parts)


def normalize_entry(entry: Dict[str, object]) -> Optional[Dict[str, str]]:
    try:
        number = str(entry["number"]).strip()
        english = str(entry["english"]).strip()
        japanese = normalize_japanese_meaning(str(entry["japanese"]).strip())
    except KeyError:
        return None

    if not number or not english or not japanese:
        return None

    return {
        "number": number,
        "english": english,
        "japanese": japanese,
    }


def parse_ocr_response(response: str) -> List[Dict[str, str]]:
    cleaned = strip_code_fence(response)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        maybe_entries = parsed.get("entries")
        if isinstance(maybe_entries, list):
            parsed = maybe_entries

    if isinstance(parsed, list):
        rows = [normalize_entry(row) for row in parsed if isinstance(row, dict)]
        return [row for row in rows if row is not None]

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return []

    if len(lines) == 1 and lines[0] == "[]":
        return []

    reader = csv.DictReader(io.StringIO(cleaned))
    rows = []
    for row in reader:
        if not row:
            continue
        normalized = normalize_entry(
            {
                "number": row.get("number", ""),
                "english": row.get("english", ""),
                "japanese": row.get("japanese", ""),
            }
        )
        if normalized is not None:
            rows.append(normalized)

    return rows


class MiniCPMVocabularyExtractor:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        prompt_path: str = "prompts.json",
        prompt_key: str = DEFAULT_PROMPT_KEY,
    ):
        self.model_id = model_id
        self.prompt = load_prompt(prompt_path, prompt_key)

        print("[Info] VLM モデルをロード中...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        print("[Info] VLM モデルのロードが完了しました。")

    def extract_from_frame(self, frame: np.ndarray) -> List[Dict[str, str]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        return self.extract_from_image(image)

    def extract_from_path(self, image_path: str) -> List[Dict[str, str]]:
        image = Image.open(image_path).convert("RGB")
        return self.extract_from_image(image)

    def extract_from_image(self, image: Image.Image) -> List[Dict[str, str]]:
        msgs = [
            {
                "role": "user",
                "content": [image, self.prompt],
            }
        ]

        response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
        )
        return parse_ocr_response(response)


def save_entries_to_csv(entries: Sequence[Dict[str, str]], output_path: str):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["number", "english", "japanese"])
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)
