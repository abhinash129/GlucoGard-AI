import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.config import DATA_RAW, DATA_PROCESSED, CNN_IMG_SIZE

CACHE_X = DATA_PROCESSED / "aptos_x.npy"
CACHE_Y = DATA_PROCESSED / "aptos_y.npy"

MAX_SKIP_FRACTION = 0.02   # refuse to cache if more than 2% of images are unreadable


def crop_black_border(img, tol=7):
    """Remove the black margin around the circular retina."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() < 0.1 * mask.size:      # image is almost all dark: leave it alone
        return img
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def preprocess_fundus(img_rgb, size=CNN_IMG_SIZE):
    img = crop_black_border(img_rgb)
    img = cv2.resize(img, size)
    # Ben Graham trick: subtract a blurred copy to remove lighting differences
    blur = cv2.GaussianBlur(img, (0, 0), size[0] / 30)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def load_image(path, size=CNN_IMG_SIZE):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"could not read image: {path}")
    return preprocess_fundus(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size)


def load_aptos(force=False):
    """Return (X uint8 [n,224,224,3], y grades 0-4). Cached after the first run.

    Images that are missing or unreadable are skipped and reported. If too many
    are unreadable (the zip was probably extracted badly), nothing is cached and
    an error is raised, so a broken dataset can never be reused by accident.
    """
    if CACHE_X.exists() and CACHE_Y.exists() and not force:
        return np.load(CACHE_X), np.load(CACHE_Y)

    found = sorted((DATA_RAW / "aptos").rglob("train.csv"))
    if not found:
        raise FileNotFoundError("train.csv not found in data/raw/aptos - download APTOS first")
    csv = found[0]
    df = pd.read_csv(csv)
    img_dir = csv.parent / "train_images"

    X_list, y_list, skipped = [], [], []
    for code, grade in tqdm(zip(df["id_code"], df["diagnosis"]),
                            total=len(df), desc="preprocessing images"):
        try:
            X_list.append(load_image(img_dir / f"{code}.png"))
            y_list.append(grade)
        except Exception:
            skipped.append(code)

    if skipped:
        print(f"WARNING: skipped {len(skipped)} unreadable images, e.g. {skipped[:5]}")

    # Safety check: do not cache a broken dataset
    if len(skipped) > MAX_SKIP_FRACTION * len(df):
        raise RuntimeError(
            f"{len(skipped)} of {len(df)} images unreadable - re-extract the zip "
            "before training. Nothing was cached.")

    X = np.stack(X_list).astype(np.uint8)
    y = np.array(y_list)

    np.save(CACHE_X, X)
    np.save(CACHE_Y, y)
    print(f"cached {len(y)} images to {CACHE_X.parent}")
    return X, y