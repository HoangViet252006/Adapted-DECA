import os
import argparse
import torch
import numpy as np
import cv2
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from glob import glob
from tqdm import tqdm

from . import FAN


def get_args():
    parser = argparse.ArgumentParser(description="Build datasets")
    parser.add_argument("--input_video_path", "-i", type=str,
                        default="TestSamples/AFLW2000/image00302.jpg")
    parser.add_argument("--is_crop", "-c", action="store_true", default=True)
    parser.add_argument("--is_lmk", "-l", action="store_true", default=True)
    parser.add_argument("--output_dir", "-o", type=str, default="My_datasets")
    return parser.parse_args()


def bbox2point(left: float, right: float, top: float, bottom: float):
    """Convert bounding box to center and scale."""
    old_size = (right - left + bottom - top) / 2 * 1.1
    center = np.array([
        right - (right - left) / 2.0,
        bottom - (bottom - top) / 2.0
    ])
    return old_size, center


class FaceDetector:
    """Wrapper to avoid reloading FAN model for each image."""
    def __init__(self, device="cuda"):
        self.model = FAN.FAN(device=device)

    def detect_and_crop(self, image: np.ndarray, crop_size=224, scale=1.25):
        h, w, _ = image.shape
        bbox, lmk = self.model.run(image)

        if bbox is None or len(bbox) < 4:
            return None

        left, right, top, bottom = bbox[0], bbox[2], bbox[1], bbox[3]
        old_size, center = bbox2point(left, right, top, bottom)
        size = int(old_size * scale)

        src_pts = np.array([
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2]
        ])
        dst_pts = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, dst_pts)

        dst_image = warp(image / 255., tform.inverse, output_shape=(crop_size, crop_size))
        dst_image = dst_image.transpose(2, 0, 1)  # [3,H,W]

        lmk_aligned = tform(lmk) if lmk is not None else None
        return {"image_cropped": torch.tensor(dst_image).float(), "lmk": lmk_aligned}


def process_image(image: np.ndarray, detector: FaceDetector):
    """Wrapper for image processing."""
    if image is None:
        return None

    return detector.detect_and_crop(image)

def process_folder(root_folder: str, is_crop=True, detector=None):
    detector = detector or FaceDetector()

    subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    for subfolder_name in tqdm(subfolders, desc="Processing subfolders"):
        subfolder_path = os.path.join(root_folder, subfolder_name)

        image_files = glob(os.path.join(subfolder_path, "*.[jp][pn]g")) + glob(os.path.join(subfolder_path, "*.bmp"))
        for image_path in tqdm(image_files, desc=subfolder_name, leave=False):
            prefix, ext = os.path.splitext(image_path)
            cropped_path = f"{prefix}_cropped.png"
            lmk_path = f"{prefix}_lmk.npy"

            if "_cropped" in image_path or os.path.exists(cropped_path):
                continue

            image = cv2.imread(image_path)
            result = process_image(image, detector)
            if result is None:
                continue

            np.save(lmk_path, result["lmk"])

            img_np = result["image_cropped"].cpu().numpy().transpose(1, 2, 0)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            cv2.imwrite(cropped_path, img_np)

def test_lmk(img_path: str, lmk_path: str, output_path: str = "landmarks_vis.png"):
    """Visualize landmarks on image and save result."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    lmk = np.load(lmk_path, allow_pickle=True)
    if lmk is None or len(lmk) == 0:
        raise ValueError(f"No landmarks found in {lmk_path}")

    # Nếu landmarks có nhiều khuôn mặt thì chỉ lấy cái đầu tiên
    if isinstance(lmk, (list, tuple)):
        lmk = lmk[0]

    for (x, y) in lmk.astype(np.int32):
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    cv2.imwrite(output_path, img)
    print(f"Saved visualization to {output_path}")


def main(args):
    detector = FaceDetector()
    image = cv2.imread(args.input_video_path)
    result = process_image(image, detector, args.is_crop)
    if result is None:
        raise RuntimeError("No face detected.")

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "landmarks.npy"), result["lmk"])

    img_np = result["image_cropped"].cpu().numpy().transpose(1, 2, 0)
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, "processed.png"), img_np)


if __name__ == '__main__':
    args = get_args()
    process_folder("dataset_DECA_cheo", True)
    # main(args)
