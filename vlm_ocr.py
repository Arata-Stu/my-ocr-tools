import ast
import csv
import io
import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_ID = "openbmb/MiniCPM-V-2_6"
DEFAULT_PROMPT_KEY = "blue_word_cards"
DEFAULT_MAX_SLICE_NUMS = 2
DEFAULT_MAX_IMAGE_SIZE = 1344
DEFAULT_PROMPT = (
    "この画像は日本の英単語帳のページです。"
    "抽出対象は、単語カード形式の見出し語だけです。"
    "各カードは、左端の縦長の番号帯、中央の大きな英単語、右側の日本語意味というレイアウトを持ちます。"
    "カラー画像では青い見出し枠に見えますが、白黒PDFでは色ではなくこのレイアウトで判定してください。"
    "見開きページに複数カードが並んでいる場合は、見えているカードをすべて抽出してください。"
    "見開き画像では、左ページの上から下、次に右ページの上から下の順で返してください。"
    "左側の帯紙・広告、上部の見出し、QRコード、音声マーク、ページ番号は無視してください。"
    "各カードから次の3項目だけ抽出してください: "
    "number=番号帯の単語番号, english=太字の英単語, japanese=日本語意味。"
    "number はカード左端の番号帯に印刷された数値をそのまま読んでください。"
    "ページ番号や問題番号やフォーマット例の数字を使ってはいけません。"
    "日本語意味が複数あるときは必ず「、」区切りで1つの文字列にまとめてください。"
    "日本語意味の中に {for} や 《to》 のような英語の補足があれば、その括弧内は削除してください。"
    "品詞記号、発音記号、例文、派生語、関連語、熟語、QRコード、ページ番号は不要です。"
    "見えていないカードを推測で補わないでください。"
    "長文ページ、会話文、subコラム、Check!!、チェックリスト、補足欄しか写っていない場合は空配列を返してください。"
    "出力はJSON配列を最優先にしてください。各要素は "
    '{"number":"1","english":"sample","japanese":"例、見本"} '
    "の形にしてください。JSONが難しい場合でも、最低限 `number,english,japanese` または `番号,英単語,日本語` のCSVとして返してください。"
    "説明文やMarkdownコードフェンスは不要です。"
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
    cleaned = re.sub(r"<\|[^|>]+\|>", "", cleaned).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def normalize_field_name(name: str) -> str:
    key = re.sub(r"[\s_\-　]", "", name.strip().lower())
    mapping = {
        "number": "number",
        "no": "number",
        "num": "number",
        "番号": "number",
        "単語番号": "number",
        "英単語番号": "number",
        "english": "english",
        "word": "english",
        "term": "english",
        "英単語": "english",
        "単語": "english",
        "見出し語": "english",
        "japanese": "japanese",
        "meaning": "japanese",
        "meanings": "japanese",
        "日本語": "japanese",
        "日本語意味": "japanese",
        "意味": "japanese",
        "和訳": "japanese",
    }
    return mapping.get(key, key)


def canonicalize_entry(raw: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in raw.items():
        normalized[normalize_field_name(str(key))] = value
    return normalized


def try_parse_json_fragment(text: str) -> Optional[object]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            fragment = text[start : end + 1]
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                continue
    return None


def try_parse_python_literal_fragment(text: str) -> Optional[object]:
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            fragment = text[start : end + 1]
            try:
                return ast.literal_eval(fragment)
            except (SyntaxError, ValueError):
                continue
    return None


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
    entry = canonicalize_entry(entry)
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


def sort_entries(entries: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    def sort_key(entry: Dict[str, str]) -> Tuple[int, object]:
        number = entry.get("number", "")
        if re.fullmatch(r"\d+", number):
            return (0, int(number))
        return (1, number)

    return sorted(entries, key=sort_key)


def dedupe_entries(entries: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    seen_numbers = set()
    unique_entries: List[Dict[str, str]] = []
    for entry in entries:
        number = entry.get("number", "")
        if not number or number in seen_numbers:
            continue
        seen_numbers.add(number)
        unique_entries.append(entry)
    return sort_entries(unique_entries)


def parse_csv_like_response(cleaned: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    candidates = [cleaned]
    if "|" in cleaned:
        markdown_lines = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.fullmatch(r"\|?[\s:\-]+\|?[\s:\-\|]*", stripped):
                continue
            if "|" in stripped:
                markdown_lines.append(stripped.strip("|"))
        if markdown_lines:
            candidates.append("\n".join(markdown_lines))

    for candidate in candidates:
        reader = csv.DictReader(io.StringIO(candidate))
        if reader.fieldnames:
            mapped = [normalize_field_name(field) for field in reader.fieldnames if field]
            if {"number", "english", "japanese"}.issubset(set(mapped)):
                for row in reader:
                    if not row:
                        continue
                    canonical_row = {}
                    for key, value in row.items():
                        if key is None:
                            continue
                        canonical_row[normalize_field_name(key)] = value or ""
                    normalized = normalize_entry(canonical_row)
                    if normalized is not None:
                        rows.append(normalized)
                if rows:
                    return rows

        # Headerless CSV lines such as `1,history,歴史、経歴`
        for delimiter in (",", "\t"):
            tmp_rows = []
            for line in candidate.splitlines():
                stripped = line.strip().strip("|")
                if not stripped:
                    continue
                parts = [part.strip() for part in stripped.split(delimiter)]
                if len(parts) < 3:
                    continue
                if not re.fullmatch(r"\d+", parts[0]):
                    continue
                number = parts[0]
                english = parts[1]
                japanese = delimiter.join(parts[2:]).strip()
                normalized = normalize_entry(
                    {
                        "number": number,
                        "english": english,
                        "japanese": japanese,
                    }
                )
                if normalized is not None:
                    tmp_rows.append(normalized)
            if tmp_rows:
                return tmp_rows

    return []


def parse_key_value_blocks(cleaned: str) -> List[Dict[str, str]]:
    rows = []
    current: Dict[str, str] = {}

    line_pattern = re.compile(
        r"^(number|english|japanese|番号|英単語|日本語|意味)\s*[:：]\s*(.+)$",
        re.IGNORECASE,
    )
    compact_pattern = re.compile(
        r"(?:number|番号)\s*[:：]\s*(?P<number>\d+).*?"
        r"(?:english|英単語)\s*[:：]\s*(?P<english>[^,，;；]+).*?"
        r"(?:japanese|日本語|意味)\s*[:：]\s*(?P<japanese>.+)$",
        re.IGNORECASE,
    )

    for line in [line.strip(" -*\t") for line in cleaned.splitlines() if line.strip()]:
        compact_match = compact_pattern.search(line)
        if compact_match:
            normalized = normalize_entry(compact_match.groupdict())
            if normalized is not None:
                rows.append(normalized)
            continue

        match = line_pattern.match(line)
        if not match:
            if current and len(current) >= 3:
                normalized = normalize_entry(current)
                if normalized is not None:
                    rows.append(normalized)
                current = {}
            continue

        key = normalize_field_name(match.group(1))
        value = match.group(2).strip()
        current[key] = value
        if {"number", "english", "japanese"}.issubset(current.keys()):
            normalized = normalize_entry(current)
            if normalized is not None:
                rows.append(normalized)
            current = {}

    if current and {"number", "english", "japanese"}.issubset(current.keys()):
        normalized = normalize_entry(current)
        if normalized is not None:
            rows.append(normalized)

    return rows


def parse_python_dict_lines(cleaned: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    # Case 1: one dict/list as a Python literal
    parsed_literal = try_parse_python_literal_fragment(cleaned)
    if isinstance(parsed_literal, dict):
        normalized = normalize_entry(parsed_literal)
        if normalized is not None:
            return [normalized]
    if isinstance(parsed_literal, list):
        literal_rows = [
            normalize_entry(item) for item in parsed_literal if isinstance(item, dict)
        ]
        literal_rows = [row for row in literal_rows if row is not None]
        if literal_rows:
            return literal_rows

    # Case 2: newline-delimited Python dict repr
    for line in cleaned.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped:
            continue
        if "{" not in stripped or "}" not in stripped:
            continue

        start = stripped.find("{")
        end = stripped.rfind("}")
        if end <= start:
            continue

        fragment = stripped[start : end + 1]
        parsed = try_parse_python_literal_fragment(fragment)
        if isinstance(parsed, dict):
            normalized = normalize_entry(parsed)
            if normalized is not None:
                rows.append(normalized)

    return rows


def parse_ocr_response(response: str) -> Tuple[List[Dict[str, str]], str]:
    cleaned = strip_code_fence(response)

    parsed = try_parse_json_fragment(cleaned)

    if isinstance(parsed, dict):
        maybe_entries = parsed.get("entries")
        if isinstance(maybe_entries, list):
            parsed = maybe_entries

    if isinstance(parsed, list):
        rows = [normalize_entry(row) for row in parsed if isinstance(row, dict)]
        normalized_rows = [row for row in rows if row is not None]
        return normalized_rows, cleaned

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return [], cleaned

    if len(lines) == 1 and lines[0] == "[]":
        return [], cleaned

    rows = parse_csv_like_response(cleaned)
    if rows:
        return rows, cleaned

    rows = parse_python_dict_lines(cleaned)
    if rows:
        return rows, cleaned

    rows = parse_key_value_blocks(cleaned)
    if rows:
        return rows, cleaned

    return [], cleaned


class MiniCPMVocabularyExtractor:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        prompt_path: str = "prompts.json",
        prompt_key: str = DEFAULT_PROMPT_KEY,
        max_slice_nums: int = DEFAULT_MAX_SLICE_NUMS,
        max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
    ):
        self.model_id = model_id
        self.prompt = load_prompt(prompt_path, prompt_key)
        self.max_slice_nums = max(1, int(max_slice_nums))
        self.max_image_size = max(448, int(max_image_size))

        print("[Info] VLM モデルをロード中...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        if not self.model_id.endswith("-int4"):
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModel.from_pretrained(
            self.model_id,
            **load_kwargs,
        ).eval()
        print("[Info] VLM モデルのロードが完了しました。")

    def extract_from_frame(self, frame: np.ndarray) -> List[Dict[str, str]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        return self.extract_from_image(image)

    def extract_from_path(self, image_path: str) -> List[Dict[str, str]]:
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        return self.extract_from_image(image)

    def _crop_relative(
        self,
        image: Image.Image,
        left_ratio: float,
        top_ratio: float,
        right_ratio: float,
        bottom_ratio: float,
    ) -> Image.Image:
        width, height = image.size
        left = max(0, min(width - 1, int(round(width * left_ratio))))
        top = max(0, min(height - 1, int(round(height * top_ratio))))
        right = max(left + 1, min(width, int(round(width * right_ratio))))
        bottom = max(top + 1, min(height, int(round(height * bottom_ratio))))
        return image.crop((left, top, right, bottom))

    def _resize_for_ocr(self, image: Image.Image, long_side_limit: int) -> Image.Image:
        width, height = image.size
        long_side = max(width, height)
        if long_side <= long_side_limit:
            return image

        scale = long_side_limit / long_side
        resized = image.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            resample=Image.Resampling.LANCZOS,
        )
        return resized

    def _build_ocr_views(self, image: Image.Image) -> List[Tuple[str, Image.Image]]:
        width, height = image.size
        aspect_ratio = width / max(height, 1)
        if aspect_ratio < 1.2:
            return [("full", image)]

        views = [
            (
                "left_page",
                self._crop_relative(image, 0.15, 0.03, 0.54, 0.98),
            ),
            (
                "right_page",
                self._crop_relative(image, 0.47, 0.03, 0.985, 0.98),
            ),
            (
                "full_spread",
                self._crop_relative(image, 0.10, 0.03, 0.985, 0.98),
            ),
        ]
        return views

    def _chat_with_image(
        self,
        image: Image.Image,
        max_slice_nums: int,
    ) -> str:
        msgs = [
            {
                "role": "user",
                "content": [image, self.prompt],
            }
        ]

        return self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            use_image_id=False,
            max_slice_nums=max_slice_nums,
        )

    def _extract_rows_and_raw_single_view(
        self,
        image: Image.Image,
        view_label: str,
    ) -> Tuple[List[Dict[str, str]], str]:
        prepared = self._resize_for_ocr(image, self.max_image_size)

        attempts = [
            (prepared, self.max_slice_nums, f"max_slice_nums={self.max_slice_nums}"),
        ]

        if self.max_slice_nums > 1:
            attempts.append((prepared, 1, "max_slice_nums=1"))

        for fallback_size in (1120, 896):
            resized = self._resize_for_ocr(prepared, fallback_size)
            if resized.size != prepared.size:
                attempts.append((resized, 1, f"max_slice_nums=1, long_side<={fallback_size}"))

        tried = set()
        for attempt_image, attempt_slices, label in attempts:
            key = (attempt_image.size, attempt_slices)
            if key in tried:
                continue
            tried.add(key)

            try:
                response = self._chat_with_image(attempt_image, attempt_slices)
                if key != (prepared.size, self.max_slice_nums):
                    print(f"[Info] {view_label}: 省メモリ設定でOCR成功: {label}")
                rows, _ = parse_ocr_response(response)
                return rows, response
            except torch.cuda.OutOfMemoryError:
                print(f"[Warn] {view_label}: CUDA OOM。{label} で再試行します。")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        raise RuntimeError(
            "CUDA OOM が解消できませんでした。"
            " より小さい画像にするか、openbmb/MiniCPM-V-2_6-int4 の利用を検討してください。"
        )

    def extract_from_image(self, image: Image.Image) -> List[Dict[str, str]]:
        base_image = ImageOps.exif_transpose(image).convert("RGB")
        rows = []
        for view_label, view_image in self._build_ocr_views(base_image):
            view_rows, _ = self._extract_rows_and_raw_single_view(view_image, view_label)
            rows.extend(view_rows)
        return dedupe_entries(rows)


def save_entries_to_csv(entries: Sequence[Dict[str, str]], output_path: str):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["number", "english", "japanese"])
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)


def extract_entries_and_raw(
    extractor: "MiniCPMVocabularyExtractor",
    image_path: str,
) -> Tuple[List[Dict[str, str]], str]:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    entries = []
    raw_sections = []

    for view_label, view_image in extractor._build_ocr_views(image):
        rows, raw = extractor._extract_rows_and_raw_single_view(view_image, view_label)
        entries.extend(rows)
        cleaned = strip_code_fence(raw)
        if cleaned:
            raw_sections.append(f"[{view_label}]\n{cleaned}")

    return dedupe_entries(entries), "\n\n".join(raw_sections)
