import cv2
import csv
import os
import numpy as np
from collections import Counter
from typing import Dict, List, Optional

class WordCardProcessor:
    def __init__(self, csv_path: str, diff_threshold: float = 5.0):
        self.csv_path = csv_path
        self.diff_threshold = diff_threshold # 差分の閾値（値が小さいほど「完全に静止」を求める）
        # 抽出したデータを一時保存する辞書 {番号: {'eng': [候補1, 候補2...], 'jpn': [候補1...]}}
        self.raw_data: Dict[int, Dict[str, List[str]]] = {} 

    def load_existing_data(self):
        """リカバリー用：既存のCSVがあれば読み込んでベースにする（Upsert）"""
        if not os.path.exists(self.csv_path):
            return

        with open(self.csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    num = int(row['番号'])
                    # 既存データは確定済みとして、多めに重み付け（ここではダミーで3回分追加）
                    if row['英単語'] and row['日本語']:
                        self.raw_data.setdefault(num, {'eng': [], 'jpn': []})
                        self.raw_data[num]['eng'].extend([row['英単語']] * 3)
                        self.raw_data[num]['jpn'].extend([row['日本語']] * 3)
                except ValueError:
                    continue
        print(f"[Info] 既存データ {self.csv_path} を読み込みました。")

    def _dummy_vlm_predict(self, frame: np.ndarray) -> List[Dict[str, str]]:
        """
        【モックアップ】ここに本来はQwen-VLなどの推論コードが入ります。
        今回はテスト用に、ランダムな結果（たまにノイズが混ざる）を返します。
        """
        import random
        results = []
        # 仮の単語データ
        base_words = {
            1: ("history", "歴史；経歴"), 2: ("tie", "つながり；を結ぶ"), 
            3: ("unite", "一体にする"), 4: ("culture", "文化；教養")
        }
        
        # 擬似的にフレームごとにばらつき（ノイズ）や抜けを発生させる
        for num, (eng, jpn) in base_words.items():
            if random.random() > 0.8: continue # 20%の確率でその番号を見逃す
            
            # 10%の確率で指で隠れた等のノイズが発生する
            extracted_eng = eng if random.random() > 0.1 else eng[:-1] + "x"
            
            results.append({"number": str(num), "english": extracted_eng, "japanese": jpn})
        return results

    def is_frame_static(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> bool:
        """フレーム間の差分を計算し、ページめくり中（動いている）か判定する"""
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = np.mean(diff)
        return mean_diff < self.diff_threshold

    def process_video(self, video_path: str, extract_fps: int = 2):
        """動画を読み込み、静止フレームをVLM（ダミー）に投げる"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] 動画 {video_path} が開けません。")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / extract_fps) # 1秒間に何回処理するか
        
        prev_gray = None
        frame_count = 0

        print("[Info] 動画の解析を開始します...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 指定間隔でフレームを処理
            if frame_count % frame_interval == 0:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 最初は差分比較できないのでスキップ
                if prev_gray is not None:
                    if self.is_frame_static(prev_gray, curr_gray):
                        print(f"  -> フレーム {frame_count} : 静止を確認。VLMで解析中...")
                        
                        # VLM推論の実行（今回はダミー）
                        vlm_results = self._dummy_vlm_predict(frame)
                        
                        # 結果を raw_data に蓄積
                        for res in vlm_results:
                            try:
                                num = int(res["number"])
                                self.raw_data.setdefault(num, {'eng': [], 'jpn': []})
                                self.raw_data[num]['eng'].append(res["english"])
                                self.raw_data[num]['jpn'].append(res["japanese"])
                            except ValueError:
                                continue
                    else:
                        pass # ページめくり等のためスキップ（デバッグ時はprintしてもよい）
                prev_gray = curr_gray
            frame_count += 1
        cap.release()
        print("[Info] 動画の解析が完了しました。")

    def resolve_and_save(self):
        """多数決によるノイズ除去、抜け番検知、CSVへの書き出し"""
        if not self.raw_data:
            print("[Warning] 保存するデータがありません。")
            return

        # 存在する番号の最小値と最大値を取得
        min_num = min(self.raw_data.keys())
        max_num = max(self.raw_data.keys())

        final_data = []

        # 抜け番検知のため、最小から最大まで順番にループ
        for num in range(min_num, max_num + 1):
            if num in self.raw_data:
                # 多数決で最頻値（最も多く認識された文字列）を採用
                best_eng = Counter(self.raw_data[num]['eng']).most_common(1)[0][0]
                best_jpn = Counter(self.raw_data[num]['jpn']).most_common(1)[0][0]
                
                # もし候補が割れすぎていたら要確認フラグを立てるロジック等もここに書けます
                status = ""
            else:
                # 抜け番の場合
                best_eng = ""
                best_jpn = ""
                status = "[要確認: 番号抜け]"
                print(f"[Alert] 番号 {num} が動画から見つかりませんでした。")

            final_data.append({
                "番号": num,
                "英単語": best_eng,
                "日本語": best_jpn,
                "備考": status
            })

        # CSVに書き出し
        with open(self.csv_path, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["番号", "英単語", "日本語", "備考"])
            writer.writeheader()
            writer.writerows(final_data)
        
        print(f"[Success] {self.csv_path} にデータを出力しました！")


# ==========================================
# 実行スクリプト
# ==========================================
if __name__ == "__main__":
    # 出力するCSVのパス
    OUTPUT_CSV = "words_list.csv"
    # テスト用の動画パス（ご自身のスマホで撮影した短い動画を置いてください）
    VIDEO_PATH = "sample.mp4" 

    processor = WordCardProcessor(OUTPUT_CSV)
    
    # 1. 既存のCSVがあれば読み込む（リカバリー機能）
    processor.load_existing_data()
    
    # 2. 動画の解析（静止検知とダミーVLM推論によるデータ蓄積）
    # ※VIDEO_PATHにファイルがないとエラーになりますが、ロジック確認のためダミー実行も可能です
    if os.path.exists(VIDEO_PATH):
        processor.process_video(VIDEO_PATH)
    else:
        print("[Warning] sample.mp4 が見つかりません。カメラ等のダミーフレームでテスト処理します。")
        # ダミーフレームを10回投げてテスト
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(10):
            res = processor._dummy_vlm_predict(dummy_frame)
            for r in res:
                num = int(r["number"])
                processor.raw_data.setdefault(num, {'eng': [], 'jpn': []})
                processor.raw_data[num]['eng'].append(r["english"])
                processor.raw_data[num]['jpn'].append(r["japanese"])

    # 3. 多数決、抜け番検知、保存
    processor.resolve_and_save()