"""
STEP 1 — Download & Clean Image Data
======================================
Run this first to build your dataset.

    python download_data.py

Uses icrawler (Bing backend) — much more reliable than DuckDuckGo
for bulk image downloads.

What you'll learn:
- Why data quality matters more than model architecture
- How to detect and remove corrupted images
- Why class balance affects training
- What a healthy dataset looks like before training

Output:
    data/
    ├── cats/    (~150 images)
    ├── dogs/    (~150 images)
    └── birds/   (~150 images)
"""

import os
import hashlib
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError

try:
    from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
except ImportError:
    raise ImportError("Run: pip install icrawler")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CATEGORIES = {
    "cats":  "cat photo",
    "dogs":  "dog photo",
    "birds": "bird photo",
}
IMAGES_PER_CLASS = 150
MIN_IMAGE_SIZE   = (64, 64)
DATA_DIR         = Path("data")


# ─────────────────────────────────────────────
# 1. DOWNLOAD IMAGES USING ICRAWLER
# ─────────────────────────────────────────────
def download_images(query: str, save_dir: Path, max_images: int) -> int:
    """
    Download images using Bing Image Search via icrawler.

    icrawler is more reliable than direct API calls because:
    - It handles pagination automatically
    - It manages request throttling internally
    - Falls back gracefully on errors
    - Supports Google, Bing, and other backends
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading: '{query}' → {save_dir}")

    crawler = BingImageCrawler(
        storage={"root_dir": str(save_dir)},
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        log_level=40,           # ERROR only — suppress verbose logs
    )

    crawler.crawl(
        keyword=query,
        max_num=max_images,
        min_size=(MIN_IMAGE_SIZE[0], MIN_IMAGE_SIZE[1]),
        file_idx_offset=0,
    )

    count = len(list(save_dir.glob("*.jpg")) +
                list(save_dir.glob("*.jpeg")) +
                list(save_dir.glob("*.png")))
    print(f"  Downloaded: {count} images")
    return count


# ─────────────────────────────────────────────
# 2. RENAME ALL FILES TO CONSISTENT FORMAT
# ─────────────────────────────────────────────
def normalize_filenames(class_dir: Path):
    """
    icrawler saves files as 000001.jpg, 000002.jpg etc.
    We rename using content hash to make deduplication reliable.
    """
    all_files = (
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.jpeg")) +
        list(class_dir.glob("*.png")) +
        list(class_dir.glob("*.JPG")) +
        list(class_dir.glob("*.JPEG")) +
        list(class_dir.glob("*.PNG"))
    )

    seen_hashes = set()
    kept = 0

    for fpath in all_files:
        try:
            data = fpath.read_bytes()
            h    = hashlib.md5(data).hexdigest()[:12]

            if h in seen_hashes:
                fpath.unlink()    # Duplicate — remove
                continue
            seen_hashes.add(h)

            new_path = class_dir / f"{h}.jpg"
            if new_path != fpath:
                fpath.rename(new_path)
            kept += 1

        except Exception:
            fpath.unlink(missing_ok=True)

    return kept


# ─────────────────────────────────────────────
# 3. POST-DOWNLOAD AUDIT — Remove bad images
# ─────────────────────────────────────────────
def audit_class(class_dir: Path) -> tuple[int, int]:
    """
    Re-scan every image after download.

    WHY AUDIT?
    ──────────
    Downloaders silently save corrupted/truncated files.
    A single bad image causes a cryptic crash mid-training.
    Better to catch everything here.

    We check for:
    1. Corrupt files  — PIL can't open them
    2. Too small      — No useful visual signal
    3. Bad aspect ratio — Banners, strips, UI screenshots
    """
    valid = 0
    removed = 0

    for fpath in list(class_dir.glob("*.jpg")):
        try:
            # PIL verify checks file integrity (not just header)
            img = Image.open(fpath)
            img.verify()

            # Re-open after verify (verify closes the file)
            img = Image.open(fpath).convert("RGB")
            w, h = img.size

            # Filter 1: Minimum size
            if w < MIN_IMAGE_SIZE[0] or h < MIN_IMAGE_SIZE[1]:
                fpath.unlink()
                removed += 1
                continue

            # Filter 2: Extreme aspect ratio (likely banners/UI)
            ratio = w / h
            if ratio > 4.0 or ratio < 0.25:
                fpath.unlink()
                removed += 1
                continue

            valid += 1

        except (UnidentifiedImageError, Exception):
            fpath.unlink(missing_ok=True)
            removed += 1

    return valid, removed


# ─────────────────────────────────────────────
# 4. DATASET AUDIT + CLASS BALANCE CHECK
# ─────────────────────────────────────────────
def audit_dataset(data_dir: Path):
    print("\n" + "="*60)
    print("POST-DOWNLOAD AUDIT")
    print("="*60)

    total_valid   = 0
    total_removed = 0
    counts = {}

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        # Normalize filenames first
        normalize_filenames(class_dir)

        # Then audit
        valid, removed = audit_class(class_dir)
        total_valid   += valid
        total_removed += removed
        counts[class_dir.name] = valid

        bar = "█" * (valid // 5)
        print(f"  {class_dir.name:10s} │ {bar:<30} {valid:3d} valid, {removed} removed")

    print(f"\n  Total valid  : {total_valid}")
    print(f"  Total removed: {total_removed}")

    # ── CLASS BALANCE WARNING ──────────────────────────────
    # Imbalanced classes bias the model toward the majority class.
    # CrossEntropyLoss treats all classes equally by default.
    # If one class has 2× more images, the model learns to predict
    # it more often just to minimise average loss.
    if counts:
        min_c = min(counts.values())
        max_c = max(counts.values())
        if min_c > 0 and max_c / min_c > 1.5:
            print(f"\n  ⚠️  CLASS IMBALANCE DETECTED (ratio {max_c/min_c:.1f}x)")
            print(f"  Consider using WeightedRandomSampler in train.py")
            print(f"  or adding more images to underrepresented classes.")
        else:
            print(f"\n  ✓ Classes are balanced")

    return total_valid, counts


def print_summary(counts: dict):
    print("\n" + "="*60)
    print("DATASET READY")
    print("="*60)
    for cls, cnt in sorted(counts.items()):
        bar = "█" * (cnt // 5)
        print(f"  {cls:10s} │ {bar} {cnt}")

    total = sum(counts.values())
    train_est = int(total * 0.8)
    val_est   = total - train_est
    print(f"\n  Estimated split (80/20):")
    print(f"    Train : ~{train_est} images")
    print(f"    Val   : ~{val_est} images")
    print(f"\nNext step: python train.py")
    print("="*60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("="*60)
    print("IMAGE CLASSIFIER — DATA DOWNLOAD + CLEANING")
    print("="*60)
    print(f"\nCategories   : {list(CATEGORIES.keys())}")
    print(f"Images/class : {IMAGES_PER_CLASS}")
    print(f"Backend      : Bing Image Search (via icrawler)")
    print(f"Output       : {DATA_DIR.resolve()}")

    for class_name, query in CATEGORIES.items():
        print(f"\n{'─'*60}")
        print(f"CLASS: {class_name.upper()}")
        download_images(query, DATA_DIR / class_name, IMAGES_PER_CLASS)

    total, counts = audit_dataset(DATA_DIR)

    if total < 60:
        print(f"\n⚠️  Only {total} images downloaded — may be too few to train well.")
        print(f"   Try running again, or reduce IMAGES_PER_CLASS and use what you have.")
    else:
        print_summary(counts)


if __name__ == "__main__":
    main()
