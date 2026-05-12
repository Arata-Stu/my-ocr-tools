import argparse
import json

from vlm_ocr import (
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_MAX_SLICE_NUMS,
    DEFAULT_MODEL_ID,
    MiniCPMVocabularyExtractor,
    save_entries_to_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="単語帳の単一画像から青枠単語だけをVLMで抽出します。"
    )
    parser.add_argument("image_path", help="入力画像パス")
    parser.add_argument(
        "--output",
        default="last_result.csv",
        help="出力CSVパス",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="使用するVLMモデルID",
    )
    parser.add_argument(
        "--prompt-path",
        default="prompts.json",
        help="抽出プロンプト定義のJSONファイル",
    )
    parser.add_argument(
        "--prompt-key",
        default="blue_word_cards",
        help="prompts.json 内のキー",
    )
    parser.add_argument(
        "--max-slice-nums",
        type=int,
        default=DEFAULT_MAX_SLICE_NUMS,
        help="MiniCPM-V に渡す画像スライス上限",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=DEFAULT_MAX_IMAGE_SIZE,
        help="OCR前に画像の長辺をこの値以下へ縮小する",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    extractor = MiniCPMVocabularyExtractor(
        model_id=args.model_id,
        prompt_path=args.prompt_path,
        prompt_key=args.prompt_key,
        max_slice_nums=args.max_slice_nums,
        max_image_size=args.max_image_size,
    )
    entries = extractor.extract_from_path(args.image_path)

    print(json.dumps(entries, ensure_ascii=False, indent=2))
    save_entries_to_csv(entries, args.output)
    print(f"[Info] {args.output} に保存しました。")


if __name__ == "__main__":
    main()
