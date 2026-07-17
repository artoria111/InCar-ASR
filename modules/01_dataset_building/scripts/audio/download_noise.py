#!/usr/bin/env python3
"""
车载噪声数据下载 — 从公开数据集获取 12 类车载噪声

数据源:
  - ESC-50:     环境声音分类数据集 (https://github.com/karolpiczak/ESC-50)
  - FSDnoisy18k: 噪声语音数据集
  - Freesound:   众包音效库 (需 API key)

Usage:
  python download_noise.py --output_dir data/raw
"""

import argparse, os, sys, urllib.request, zipfile, tarfile, shutil
from pathlib import Path
import csv

# ESC-50: 2000 environmental sounds, 50 classes, 5s each
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

# ESC-50 类别 → 车载噪声类别映射
ESC50_CAR_MAPPING = {
    "engine": "engine_idle",          # 发动机
    "engine_idling": "engine_idle",
    "car_horn": "horn_and_traffic",
    "car_alarm": "horn_and_traffic",
    "train": "horn_and_traffic",
    "airplane": "wind_noise",
    "helicopter": "wind_noise",
    "wind": "wind_noise",
    "rain": "rain",
    "thunderstorm": "rain",
    "pouring_water": "rain",
    "crackling_fire": "cabin_music",
    "door_wood_creaks": "window_open",
    "siren": "horn_and_traffic",
    "street_music": "cabin_music",
    "vacuum_cleaner": "air_conditioner",
    "water_tap": "rain",
}


def download_esc50(output_dir: Path) -> int:
    """下载并解压 ESC-50 数据集，提取车载相关噪声"""
    print("[ESC-50] Downloading...")
    zip_path = output_dir / "esc50.zip"

    if not zip_path.exists():
        urllib.request.urlretrieve(ESC50_URL, zip_path)
        print(f"  Downloaded: {zip_path}")

    extract_dir = output_dir / "ESC-50-master"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        print(f"  Extracted: {extract_dir}")

    # 读取 ESC-50 元数据
    meta_path = extract_dir / "meta" / "esc50.csv"
    if not meta_path.exists():
        print("  WARNING: ESC-50 metadata not found. Please download manually.")
        print("  URL: https://github.com/karolpiczak/ESC-50")
        return 0

    audio_dir = extract_dir / "audio"
    count = 0

    with open(meta_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row.get("category", "").lower().replace(" ", "_")
            if category in ESC50_CAR_MAPPING:
                car_cat = ESC50_CAR_MAPPING[category]
                src = audio_dir / row["filename"]
                dst = output_dir / "raw" / car_cat / row["filename"]
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    count += 1

    print(f"  Copied {count} files to {output_dir / 'raw'}")
    return count


def create_noise_index(output_dir: Path):
    """生成噪声文件索引"""
    raw_dir = output_dir / "raw"
    if not raw_dir.exists():
        return

    index = []
    for cat_dir in sorted(raw_dir.iterdir()):
        if cat_dir.is_dir():
            for audio_file in cat_dir.rglob("*.wav"):
                index.append({
                    "category": cat_dir.name,
                    "file": str(audio_file.relative_to(output_dir)),
                    "source": "esc50",
                    "duration_s": "",
                    "license": "CC BY-NC 4.0",
                })
            for audio_file in cat_dir.rglob("*.flac"):
                index.append({
                    "category": cat_dir.name,
                    "file": str(audio_file.relative_to(output_dir)),
                    "source": "esc50",
                    "duration_s": "",
                    "license": "CC BY-NC 4.0",
                })

    index_path = output_dir / "noise_index.csv"
    with open(index_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["category", "file", "source", "duration_s", "license"])
        writer.writeheader()
        writer.writerows(index)
    print(f"[Index] Written {len(index)} entries to {index_path}")


def parse_args():
    p = argparse.ArgumentParser(description="下载车载噪声数据")
    p.add_argument("--output_dir", default="data", help="输出目录")
    p.add_argument("--source", choices=["esc50", "all"], default="all")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    if args.source in ("esc50", "all"):
        try:
            total += download_esc50(output_dir)
        except Exception as e:
            print(f"[ESC-50] Failed: {e}")
            print("  Manual download: https://github.com/karolpiczak/ESC-50")
            print("  Place extracted audio/ in data/raw/<category>/")

    create_noise_index(output_dir)
    print(f"\nDone. Total noise files: {total}")
    if total == 0:
        print("\nNo automatic downloads succeeded. Please manually download:")
        print("  1. ESC-50: https://github.com/karolpiczak/ESC-50")
        print("  2. FSDnoisy18k: https://zenodo.org/record/2529934")
        print("  3. Freesound: https://freesound.org (search: engine idle, wind, rain, traffic)")


if __name__ == "__main__":
    main()
