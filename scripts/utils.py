import os
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder


def list_images_and_labels(root_dir, classes=None):
    """Return lists of image file paths and labels found under root_dir."""
    root = Path(root_dir)
    paths, labels = [], []
    for cls in sorted(os.listdir(root)):
        cls_path = root / cls
        if not cls_path.is_dir():
            continue
        if classes and cls not in classes:
            continue
        for imgfn in cls_path.glob('*'):
            if imgfn.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                paths.append(str(imgfn))
                labels.append(cls)
    return paths, labels


def encode_labels(labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le
