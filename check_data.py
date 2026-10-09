import os, yaml
from collections import Counter

DATA = "D:/Computer_vision/archive"

with open(f"{DATA}/data.yaml") as f:
    cfg = yaml.safe_load(f)
print("Classes:", cfg["names"])

for split in ["train", "val"]:
    img_dir = f"{DATA}/{split}/images"
    lbl_dir = f"{DATA}/{split}/labels"
    imgs = os.listdir(img_dir)
    counts = Counter()
    for lbl in os.listdir(lbl_dir):
        if lbl.endswith(".txt"):
            with open(f"{lbl_dir}/{lbl}") as f:
                for line in f:
                    if line.strip():
                        counts[int(line.split()[0])] += 1
    print(split, "| images:", len(imgs), "| boxes per class:", dict(counts))