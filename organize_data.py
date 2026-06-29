"""
Organize downloaded Kaggle datasets into the project data folders.
Run this after both APTOS and Kermany OCT downloads complete.

Usage:
    python organize_data.py
"""

import os
import shutil
import csv
from pathlib import Path

HOME = os.path.expanduser("~")
DR_DIR = os.path.join(HOME, "DR")
DOWNLOADS = os.path.join(DR_DIR, "downloads")
DATA_DIR = os.path.join(DR_DIR, "data")

OCT_DEST = os.path.join(DATA_DIR, "OCT")
FUNDUS_DEST = os.path.join(DATA_DIR, "Fundus")


def organize_aptos():
    """
    APTOS 2019 dataset structure (after unzip):
      downloads/aptos/
        train_images/ or train/  — contains .png files
        train.csv                — id_code, diagnosis (0-4)
    
    We copy:
      diagnosis 0       → NOT used (healthy fundus)
      diagnosis 1-4     → ~/DR/data/Fundus/  (DR positive)
    
    For healthy eyes we also include diagnosis 0 in OCT folder? No.
    Actually for this project:
      OCT folder  = healthy (label 0)
      Fundus      = DR detected (label 1)
    
    So we put DR-positive fundus images (grades 1-4) into Fundus folder.
    And we can also put grade-0 (healthy) fundus into OCT folder as "healthy" samples,
    OR keep OCT images as healthy and fundus DR images as diseased.
    
    Let's keep it simple and match the original project design:
      - OCT folder gets NORMAL OCT images (healthy)
      - Fundus folder gets DR-positive fundus images
    """
    print("\n📁 Organizing APTOS 2019 (Fundus) dataset...")
    
    aptos_dir = os.path.join(DOWNLOADS, "aptos")
    
    # Find the CSV file
    csv_path = None
    for root, dirs, files in os.walk(aptos_dir):
        for f in files:
            if f.endswith(".csv") and "train" in f.lower():
                csv_path = os.path.join(root, f)
                break
        if csv_path:
            break
    
    if not csv_path:
        print("  ⚠️  Could not find train.csv in APTOS download")
        # Fallback: just copy all images
        img_count = 0
        for root, dirs, files in os.walk(aptos_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    shutil.copy2(os.path.join(root, f), os.path.join(FUNDUS_DEST, f))
                    img_count += 1
        print(f"  Copied {img_count} images to Fundus folder (no CSV filtering)")
        return
    
    print(f"  Found CSV: {csv_path}")
    
    # Find image directory
    img_dir = None
    for candidate in ["train_images", "train", "images"]:
        for root, dirs, files in os.walk(aptos_dir):
            if candidate in dirs:
                img_dir = os.path.join(root, candidate)
                break
        if img_dir:
            break
    
    # If no specific folder found, search for images anywhere
    if not img_dir:
        img_dir = aptos_dir
    
    print(f"  Image directory: {img_dir}")
    
    # Read CSV and copy DR-positive images (grades 1-4) to Fundus
    dr_count = 0
    healthy_count = 0
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            diagnosis = int(row.get("diagnosis", row.get("level", -1)))
            img_id = row.get("id_code", row.get("image", ""))
            
            # Try to find the image file
            img_found = None
            for ext in [".png", ".jpg", ".jpeg", ""]:
                candidate_path = os.path.join(img_dir, img_id + ext)
                if os.path.exists(candidate_path):
                    img_found = candidate_path
                    break
            
            # Also search recursively
            if not img_found:
                for root, _, files in os.walk(img_dir):
                    for f_name in files:
                        if f_name.startswith(img_id):
                            img_found = os.path.join(root, f_name)
                            break
                    if img_found:
                        break
            
            if not img_found:
                continue
            
            if diagnosis >= 1:
                # DR positive → Fundus folder
                dest = os.path.join(FUNDUS_DEST, os.path.basename(img_found))
                shutil.copy2(img_found, dest)
                dr_count += 1
            else:
                healthy_count += 1
    
    print(f"  ✅ Copied {dr_count} DR-positive fundus images to Fundus folder")
    print(f"  ℹ️  Skipped {healthy_count} healthy fundus images (grade 0)")


def organize_oct():
    """
    Kermany OCT dataset structure (after unzip):
      downloads/oct/
        OCT2017/ or similar
          train/
            NORMAL/
            CNV/
            DME/
            DRUSEN/
          test/
            NORMAL/
            ...
    
    We copy NORMAL images → ~/DR/data/OCT/ (as healthy, label 0)
    """
    print("\n📁 Organizing Kermany OCT dataset...")
    
    oct_dir = os.path.join(DOWNLOADS, "oct")
    
    normal_count = 0
    
    # Walk through looking for NORMAL folders
    for root, dirs, files in os.walk(oct_dir):
        folder_name = os.path.basename(root).upper()
        if folder_name == "NORMAL":
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".bmp")):
                    shutil.copy2(
                        os.path.join(root, f),
                        os.path.join(OCT_DEST, f),
                    )
                    normal_count += 1
    
    if normal_count == 0:
        # Fallback: list what's in the download
        print("  ⚠️  No NORMAL folder found. Contents of download:")
        for item in os.listdir(oct_dir)[:20]:
            print(f"    {item}")
    else:
        print(f"  ✅ Copied {normal_count} NORMAL OCT images to OCT folder")


def main():
    # Remove old sample images
    for folder in [OCT_DEST, FUNDUS_DEST]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.startswith(("oct_sample_", "fundus_sample_")):
                    os.remove(os.path.join(folder, f))
    
    os.makedirs(OCT_DEST, exist_ok=True)
    os.makedirs(FUNDUS_DEST, exist_ok=True)
    
    # Check downloads exist
    aptos_exists = os.path.exists(os.path.join(DOWNLOADS, "aptos"))
    oct_exists = os.path.exists(os.path.join(DOWNLOADS, "oct"))
    
    if not aptos_exists and not oct_exists:
        print("❌ No downloads found. Please download datasets first.")
        return
    
    if aptos_exists:
        organize_aptos()
    else:
        print("⚠️  APTOS download not found, skipping Fundus organization")
    
    if oct_exists:
        organize_oct()
    else:
        print("⚠️  OCT download not found, skipping OCT organization")
    
    # Summary
    oct_count = len([f for f in os.listdir(OCT_DEST) if not f.startswith(".")])
    fundus_count = len([f for f in os.listdir(FUNDUS_DEST) if not f.startswith(".")])
    
    print(f"\n{'='*50}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"  OCT (Healthy):     {oct_count} images")
    print(f"  Fundus (DR):       {fundus_count} images")
    print(f"  Total:             {oct_count + fundus_count} images")
    print(f"\nNow run: python train_model.py")


if __name__ == "__main__":
    main()
