#!/usr/bin/env python3
"""
tools/generate_tangent_field.py

Offline Ground Truth Tangent Field Generator for Crack500 dataset.
Uses Double-Angle Representation:
    theta = arctan2(vh[0, 1], vh[0, 0])
    Vx = cos(2 * theta), Vy = sin(2 * theta)
This formulation is strictly invariant to the arbitrary sign assignment of eigenvectors.

Output:
    Saves .npz files containing 'vx' and 'vy' (float16) in a target directory (e.g. 'tangents/').
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from tqdm import tqdm


def compute_tangent_field_double_angle(mask: np.ndarray, patch_radius: int = 3):
    """
    Computes double-angle unit tangent field for crack mask.
    Returns:
        Vx_full: cos(2 * theta) for crack pixels, 0 for background.
        Vy_full: sin(2 * theta) for crack pixels, 0 for background.
    """
    skel = skeletonize(mask > 0)
    ys, xs = np.where(skel)
    
    H, W = mask.shape[:2]
    Vx = np.zeros((H, W), dtype=np.float32)
    Vy = np.zeros((H, W), dtype=np.float32)

    if len(xs) == 0:
        return Vx, Vy

    # Compute orientation at skeleton points via local window SVD
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - patch_radius), min(H, y + patch_radius + 1)
        x0, x1 = max(0, x - patch_radius), min(W, x + patch_radius + 1)
        patch_ys, patch_xs = np.where(skel[y0:y1, x0:x1])
        if len(patch_xs) < 2:
            continue
        pts = np.stack([patch_xs, patch_ys], axis=1).astype(np.float32)
        pts -= pts.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(pts)
            # Principal tangent direction vh[0] = [dx, dy]
            theta = np.arctan2(vh[0, 1], vh[0, 0])
            # Double-angle representation (strictly sign-invariant)
            Vx[y, x] = np.cos(2.0 * theta)
            Vy[y, x] = np.sin(2.0 * theta)
        except Exception:
            continue

    # Propagate to all crack body pixels via nearest skeleton point
    if skel.any():
        _, indices = distance_transform_edt(~skel, return_indices=True)
        crack_mask = (mask > 0).astype(np.float32)
        Vx_full = Vx[indices[0], indices[1]] * crack_mask
        Vy_full = Vy[indices[0], indices[1]] * crack_mask
    else:
        Vx_full = Vx
        Vy_full = Vy

    return Vx_full, Vy_full


def process_single_mask(args_tuple):
    mask_path, out_dir = args_tuple
    base_name = os.path.splitext(os.path.basename(mask_path))[0]
    out_path = os.path.join(out_dir, base_name + ".npz")
    
    if os.path.exists(out_path):
        return True

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False

    vx, vy = compute_tangent_field_double_angle(mask)
    np.savez_compressed(out_path, vx=vx.astype(np.float16), vy=vy.astype(np.float16))
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate Double-Angle Tangent Field GT for Crack500")
    parser.add_argument("--mask-dir", type=str, default="datasets/Crack500_ready/train/masks",
                        help="Path to masks folder")
    parser.add_argument("--out-dir", type=str, default="datasets/Crack500_ready/train/tangents",
                        help="Path to output tangents folder")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "*.png")) + glob.glob(os.path.join(args.mask_dir, "*.jpg")))
    print(f"Found {len(mask_files)} mask files in {args.mask_dir}")
    if len(mask_files) == 0:
        print("No files found. Exiting.")
        return

    tasks = [(p, args.out_dir) for p in mask_files]
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(process_single_mask, tasks), total=len(tasks), desc="Generating Tangent GT"))

    elapsed = time.time() - t0
    success = sum(results)
    print(f"Successfully processed {success}/{len(mask_files)} tangent files in {elapsed:.1f}s.")
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
