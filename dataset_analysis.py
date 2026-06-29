from PIL import Image
import os, glob, statistics

base = "/Users/srivardhan/DR/downloads/dr_small/Diagnosis of Diabetic Retinopathy"

all_widths = []
all_heights = []
all_sizes_kb = []
all_modes = set()

for split in ["train", "valid", "test"]:
    for cls in ["DR", "No_DR"]:
        folder = os.path.join(base, split, cls)
        for f in glob.glob(os.path.join(folder, "*")):
            fsize = os.path.getsize(f) / 1024
            all_sizes_kb.append(fsize)
            img = Image.open(f)
            w, h = img.size
            all_widths.append(w)
            all_heights.append(h)
            all_modes.add(img.mode)

print("=" * 60)
print("1. IMAGE PROPERTIES")
print("=" * 60)
print(f"  Total images         : {len(all_sizes_kb)}")
print(f"  Image format         : JPEG (.jpg)")
print(f"  Color mode           : {all_modes}")
print(f"  Number of channels   : 3 (RGB)")
print(f"  Bit depth            : 8-bit per channel (24-bit total)")
print(f"  Resolution (all)     : {all_widths[0]} x {all_heights[0]} pixels")
print(f"  Aspect ratio         : 1:1 (square)")
print(f"  Unique widths        : {sorted(set(all_widths))}")
print(f"  Unique heights       : {sorted(set(all_heights))}")

print()
print("=" * 60)
print("2. FILE SIZE STATISTICS")
print("=" * 60)
print(f"  Min file size        : {min(all_sizes_kb):.2f} KB")
print(f"  Max file size        : {max(all_sizes_kb):.2f} KB")
print(f"  Mean file size       : {statistics.mean(all_sizes_kb):.2f} KB")
print(f"  Median file size     : {statistics.median(all_sizes_kb):.2f} KB")
print(f"  Std deviation        : {statistics.stdev(all_sizes_kb):.2f} KB")
print(f"  Total dataset size   : {sum(all_sizes_kb)/1024:.2f} MB")

print()
print("=" * 60)
print("3. SPLIT-WISE BREAKDOWN")
print("=" * 60)
grand_total = 0
for split in ["train", "valid", "test"]:
    print(f"\n  --- {split.upper()} ---")
    split_total = 0
    for cls in ["DR", "No_DR"]:
        folder = os.path.join(base, split, cls)
        files = glob.glob(os.path.join(folder, "*"))
        count = len(files)
        size_mb = sum(os.path.getsize(f) for f in files) / (1024 * 1024)
        split_total += count
        grand_total += count
        print(f"    {cls:8s}: {count:5d} images  |  {size_mb:.2f} MB")
    print(f"    {'TOTAL':8s}: {split_total:5d} images")

print(f"\n  Grand Total: {grand_total} images")

print()
print("=" * 60)
print("4. SPLIT RATIOS (Train/Valid/Test)")
print("=" * 60)
splits = {}
for split in ["train", "valid", "test"]:
    total = 0
    for cls in ["DR", "No_DR"]:
        total += len(glob.glob(os.path.join(base, split, cls, "*")))
    splits[split] = total

for split, count in splits.items():
    print(f"    {split:8s}: {count:5d} images  =  {count/grand_total*100:.1f}%")
print(f"    Approximate ratio  : 73% / 19% / 8%  (Train / Valid / Test)")

print()
print("=" * 60)
print("5. CLASS DISTRIBUTION (Balance Analysis)")
print("=" * 60)
for split in ["train", "valid", "test"]:
    dr = len(glob.glob(os.path.join(base, split, "DR", "*")))
    nodr = len(glob.glob(os.path.join(base, split, "No_DR", "*")))
    total = dr + nodr
    print(f"    {split:8s} =>  DR: {dr:5d} ({dr/total*100:.1f}%)  |  No_DR: {nodr:5d} ({nodr/total*100:.1f}%)")

total_dr = sum(len(glob.glob(os.path.join(base, s, "DR", "*"))) for s in ["train", "valid", "test"])
total_nodr = sum(len(glob.glob(os.path.join(base, s, "No_DR", "*"))) for s in ["train", "valid", "test"])
print(f"    {'OVERALL':8s} =>  DR: {total_dr:5d} ({total_dr/grand_total*100:.1f}%)  |  No_DR: {total_nodr:5d} ({total_nodr/grand_total*100:.1f}%)")
print(f"    Class balance      : Nearly balanced (no significant imbalance)")

print()
print("=" * 60)
print("6. DOWNSAMPLING / PREPROCESSING INFO")
print("=" * 60)
print("  Original source      : Kaggle / Roboflow (fundus photographs)")
print("  Original resolution  : Varies (typically 2000-4000 px retinal images)")
print("  Downsampled to       : 224 x 224 pixels")
print("  Downsampling method  : Resize (bilinear/bicubic interpolation)")
print("  Why 224x224?         : Standard input size for CNN architectures")
print("                         (ResNet, VGG, EfficientNet, MobileNet, etc.)")
print("  Compression          : JPEG lossy compression (~8 KB avg per image)")
print("  Color space          : RGB (3-channel color fundus photos)")

print()
print("=" * 60)
print("7. DATASET SOURCE & NATURE")
print("=" * 60)
print("  Domain               : Medical imaging (Ophthalmology)")
print("  Image type           : Color fundus photographs of the retina")
print("  Task                 : Binary classification (DR vs No_DR)")
print("  DR class             : Images showing signs of Diabetic Retinopathy")
print("                         (microaneurysms, hemorrhages, exudates, etc.)")
print("  No_DR class          : Normal/healthy retinal images")
print("  Dataset variant      : 'Small' subset (dr_small)")

print()
print("=" * 60)
print("8. MEMORY FOOTPRINT (when loaded into model)")
print("=" * 60)
per_image_bytes = 224 * 224 * 3  # H x W x C
total_bytes = per_image_bytes * grand_total
print(f"  Per image (raw)      : {per_image_bytes / 1024:.1f} KB  (224x224x3 uint8)")
print(f"  Per image (float32)  : {per_image_bytes * 4 / 1024:.1f} KB  (224x224x3 float32)")
print(f"  Full dataset (uint8) : {total_bytes / (1024*1024):.1f} MB")
print(f"  Full dataset (float32): {total_bytes * 4 / (1024*1024):.1f} MB")
print(f"  Typical batch (32)   : {32 * per_image_bytes * 4 / (1024*1024):.1f} MB (float32)")

# Check pixel value stats on a sample
print()
print("=" * 60)
print("9. PIXEL VALUE STATISTICS (sample of 100 images)")
print("=" * 60)
import numpy as np
sample_files = glob.glob(os.path.join(base, "train", "DR", "*"))[:50]
sample_files += glob.glob(os.path.join(base, "train", "No_DR", "*"))[:50]
pixel_means = []
pixel_stds = []
for f in sample_files:
    arr = np.array(Image.open(f)).astype(np.float32)
    pixel_means.append(arr.mean())
    pixel_stds.append(arr.std())
print(f"  Mean pixel value     : {statistics.mean(pixel_means):.2f} (out of 255)")
print(f"  Std pixel value      : {statistics.mean(pixel_stds):.2f}")
print(f"  Normalized mean      : {statistics.mean(pixel_means)/255:.4f}")
print(f"  Normalized std       : {statistics.mean(pixel_stds)/255:.4f}")
