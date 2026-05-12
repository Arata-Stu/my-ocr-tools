import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from typing import Dict, List, Optional

import cv2
import numpy as np

from vlm_ocr import (
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_MAX_SLICE_NUMS,
    DEFAULT_MODEL_ID,
    MiniCPMVocabularyExtractor,
)


class WordCardProcessor:
    def __init__(
        self,
        csv_path: str,
        extractor: Optional[MiniCPMVocabularyExtractor] = None,
        diff_threshold: float = 5.0,
        min_static_samples: int = 2,
        max_ocr_per_static_segment: int = 2,
    ):
        self.csv_path = csv_path
        self.extractor = extractor
        self.diff_threshold = diff_threshold
        self.min_static_samples = max(1, min_static_samples)
        self.max_ocr_per_static_segment = max(1, max_ocr_per_static_segment)
        self.raw_data: Dict[int, Dict[str, List[str]]] = {}

    def load_existing_data(self):
        """リカバリー用：既存のCSVがあれば読み込んでベースにする（Upsert）"""
        if not os.path.exists(self.csv_path):
            return

        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    num = int(row["番号"])
                except (KeyError, TypeError, ValueError):
                    continue

                eng = (row.get("英単語") or "").strip()
                jpn = (row.get("日本語") or row.get("日本語意味") or "").strip()
                if not eng or not jpn:
                    continue

                self.raw_data.setdefault(num, {"eng": [], "jpn": []})
                self.raw_data[num]["eng"].extend([eng] * 3)
                self.raw_data[num]["jpn"].extend([jpn] * 3)

        print(f"[Info] 既存データ {self.csv_path} を読み込みました。")

    def _dummy_vlm_predict(self, frame: np.ndarray) -> List[Dict[str, str]]:
        """
        VLM未使用時のスモークテスト用。
        """
        import random

        results = []
        base_words = {
            1: ("history", "歴史；経歴"),
            2: ("tie", "つながり；ネクタイ；同点；を結ぶ；をつなぐ"),
            3: ("unite", "一体にする；一つにする"),
            4: ("culture", "文化；教養；培養"),
        }

        for num, (eng, jpn) in base_words.items():
            if random.random() > 0.8:
                continue

            extracted_eng = eng if random.random() > 0.1 else eng[:-1] + "x"
            results.append(
                {
                    "number": str(num),
                    "english": extracted_eng,
                    "japanese": jpn,
                }
            )

        return results

    def is_frame_static(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> bool:
        """フレーム間の差分を計算し、ページめくり中（動いている）か判定する"""
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = np.mean(diff)
        return mean_diff < self.diff_threshold

    def analyze_frame(self, frame: np.ndarray, sample_label: str):
        try:
            if self.extractor is None:
                vlm_results = self._dummy_vlm_predict(frame)
            else:
                vlm_results = self.extractor.extract_from_frame(frame)
        except Exception as exc:
            print(f"  -> {sample_label} : OCR失敗 ({exc})")
            return

        if not vlm_results:
            print(f"  -> {sample_label} : 対象の単語カードなし")
            return

        print(f"  -> {sample_label} : {len(vlm_results)} 件の候補を取得")

        for res in vlm_results:
            try:
                num = int(str(res["number"]).strip())
            except (KeyError, TypeError, ValueError):
                continue

            eng = str(res.get("english", "")).strip()
            jpn = str(res.get("japanese", "")).strip()
            if not eng or not jpn:
                continue

            self.raw_data.setdefault(num, {"eng": [], "jpn": []})
            self.raw_data[num]["eng"].append(eng)
            self.raw_data[num]["jpn"].append(jpn)

    def process_video(self, video_path: str, extract_fps: float = 2.0):
        """動画を読み込み、静止ページに対してVLM OCRを実行する"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] 動画 {video_path} が開けません。")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(fps / max(extract_fps, 0.1))))

        prev_gray = None
        frame_count = 0
        static_streak = 0
        ocr_runs_current_segment = 0

        print("[Info] 動画の解析を開始します...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    if self.is_frame_static(prev_gray, curr_gray):
                        static_streak += 1
                        if (
                            static_streak >= self.min_static_samples
                            and ocr_runs_current_segment
                            < self.max_ocr_per_static_segment
                        ):
                            print(
                                f"  -> フレーム {frame_count} : 静止を確認。OCRで解析中..."
                            )
                            self.analyze_frame(frame, f"フレーム {frame_count}")
                            ocr_runs_current_segment += 1
                    else:
                        static_streak = 0
                        ocr_runs_current_segment = 0

                prev_gray = curr_gray

            frame_count += 1

        cap.release()
        print("[Info] 動画の解析が完了しました。")

    def process_pdf(self, pdf_path: str, dpi: int = 200):
        """PDFをページ画像に変換し、各ページをVLM OCRに投入する"""
        gs_path = shutil.which("gs")
        if not gs_path:
            print("[Error] Ghostscript (gs) が見つかりません。PDFを処理できません。")
            return

        with tempfile.TemporaryDirectory(prefix="ocr_pdf_") as tmp_dir:
            output_pattern = os.path.join(tmp_dir, "page-%04d.png")
            command = [
                gs_path,
                "-q",
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=pnggray",
                f"-r{dpi}",
                f"-sOutputFile={output_pattern}",
                pdf_path,
            ]

            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
                print(f"[Error] PDFの画像化に失敗しました: {stderr or exc}")
                return

            page_files = sorted(
                [
                    os.path.join(tmp_dir, name)
                    for name in os.listdir(tmp_dir)
                    if name.lower().endswith(".png")
                ]
            )
            if not page_files:
                print("[Warning] PDFからページ画像を生成できませんでした。")
                return

            print(f"[Info] PDFの解析を開始します... ({len(page_files)} ページ)")
            for page_index, page_file in enumerate(page_files, start=1):
                page_image = cv2.imread(page_file, cv2.IMREAD_COLOR)
                if page_image is None:
                    print(f"  -> ページ {page_index} : 画像読み込み失敗")
                    continue
                self.analyze_frame(page_image, f"ページ {page_index}")
            print("[Info] PDFの解析が完了しました。")

    def resolve_and_save(self):
        """多数決によるノイズ除去、抜け番検知、CSVへの書き出し"""
        if not self.raw_data:
            print("[Warning] 保存するデータがありません。")
            return

        min_num = min(self.raw_data.keys())
        max_num = max(self.raw_data.keys())
        final_data = []

        for num in range(min_num, max_num + 1):
            if num in self.raw_data:
                best_eng = Counter(self.raw_data[num]["eng"]).most_common(1)[0][0]
                best_jpn = Counter(self.raw_data[num]["jpn"]).most_common(1)[0][0]
                status = ""
            else:
                best_eng = ""
                best_jpn = ""
                status = "[要確認: 番号抜け]"
                print(f"[Alert] 番号 {num} が動画から見つかりませんでした。")

            final_data.append(
                {
                    "番号": num,
                    "英単語": best_eng,
                    "日本語": best_jpn,
                    "備考": status,
                }
            )

        with open(self.csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["番号", "英単語", "日本語", "備考"]
            )
            writer.writeheader()
            writer.writerows(final_data)

        print(f"[Success] {self.csv_path} にデータを出力しました。")


def parse_args():
    parser = argparse.ArgumentParser(
        description="単語帳の動画またはPDFから単語番号・英単語・日本語をCSV化します。"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="sample.mp4",
        help="入力ファイルパス（動画またはPDF）",
    )
    parser.add_argument(
        "--output",
        default="words_list.csv",
        help="出力CSVパス",
    )
    parser.add_argument(
        "--extract-fps",
        type=float,
        default=2.0,
        help="1秒あたり何回サンプリングするか",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=5.0,
        help="静止判定の差分閾値",
    )
    parser.add_argument(
        "--min-static-samples",
        type=int,
        default=2,
        help="何サンプル連続で静止ならOCRするか",
    )
    parser.add_argument(
        "--max-ocr-per-static-segment",
        type=int,
        default=2,
        help="同じ静止ページに対して何回までOCRするか",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=200,
        help="PDFを画像化するときのDPI",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="使用するVLMモデルID",
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
        "--dummy",
        action="store_true",
        help="VLMを使わずダミーOCRで動作確認する",
    )
    return parser.parse_args()


def infer_input_type(input_path: str) -> str:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    return "video"


def main():
    args = parse_args()

    extractor = None
    if not args.dummy:
        extractor = MiniCPMVocabularyExtractor(
            model_id=args.model_id,
            prompt_path=args.prompt_path,
            prompt_key=args.prompt_key,
            max_slice_nums=args.max_slice_nums,
            max_image_size=args.max_image_size,
        )

    processor = WordCardProcessor(
        csv_path=args.output,
        extractor=extractor,
        diff_threshold=args.diff_threshold,
        min_static_samples=args.min_static_samples,
        max_ocr_per_static_segment=args.max_ocr_per_static_segment,
    )
    processor.load_existing_data()

    if os.path.exists(args.input_path):
        input_type = infer_input_type(args.input_path)
        if input_type == "pdf":
            processor.process_pdf(args.input_path, dpi=args.pdf_dpi)
        else:
            processor.process_video(args.input_path, extract_fps=args.extract_fps)
        processor.resolve_and_save()
        return

    if args.dummy:
        print(
            f"[Warning] {args.input_path} が見つかりません。ダミーフレームでスモークテストします。"
        )
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for index in range(10):
            processor.analyze_frame(dummy_frame, sample_label=f"dummy {index + 1}")
        processor.resolve_and_save()
        return

    print(f"[Error] 入力ファイルが見つかりません: {args.input_path}")


if __name__ == "__main__":
    main()
