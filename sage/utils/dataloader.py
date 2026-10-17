import os
import random
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

Image.MAX_IMAGE_PIXELS = None 


def pad_to_square(image: np.ndarray, mask: np.ndarray, min_size: int = 448):
    """
    Dynamic Pad-to-Square based on actual image dimensions (for DeepCrack protocol):
        target = max(H, W, min_size)
    Pads symmetrically with BORDER_CONSTANT (0) to target x target, preserving aspect ratio.
    """
    H, W = image.shape[:2]
    target = max(H, W, min_size)
    pad_h = target - H
    pad_w = target - W
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    padded_mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_img, padded_mask


def get_transformations(img_size):
    """
    SOTA Resize Protocol: Direct resize to (img_size, img_size) with nearest mask interpolation.
    Zero padding, zero artificial boundary reflection.
    """
    base_crop = [
        A.Resize(img_size, img_size, mask_interpolation=cv2.INTER_NEAREST)
    ]
    val_crop = [
        A.Resize(img_size, img_size, mask_interpolation=cv2.INTER_NEAREST)
    ]

    aug_list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    
    train_transform = A.Compose(base_crop + aug_list)
    val_transforms = A.Compose(val_crop + [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transforms


class ConfigurableMedicalDataset(Dataset):
    """
    Canonical Dataset Loader reading from a YAML configuration file.
    Supports pre-split datasets (train/val/test in separate folders).
    Clean single-path execution without redundant filters or dead code.
    """
    def __init__(self, config_path_or_dict, split='train', image_size=448, transform=None):
        self.split = split
        self.image_size = image_size
        
        # 1. Load Configuration
        if isinstance(config_path_or_dict, str):
            with open(config_path_or_dict, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_path_or_dict

        self.root_dir = self.config.get('root_dir', '')
        root_dir = self.root_dir
        
        # Auto-detect dataset domain
        self.is_crack500 = 'Crack500' in root_dir or 'crack500' in root_dir.lower()
        
        # 2. Get paths for the specific split
        if split not in self.config:
            raise ValueError(f"Split '{split}' not found in configuration file.")
            
        split_config = self.config[split]
        img_dir = split_config['images']
        mask_dir = split_config['masks']
        
        if root_dir and not os.path.isabs(img_dir):
            img_dir = os.path.join(root_dir, img_dir)
        if root_dir and not os.path.isabs(mask_dir):
            mask_dir = os.path.join(root_dir, mask_dir)
            
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Image dir '{img_dir}' or Mask dir '{mask_dir}' does not exist.")

        # 3. Settings & Masks Indexing
        self.mask_suffix = self.config.get('mask_suffix', '')
        valid_img_exts = set(['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])
        
        self.samples = []
        mask_map = {}
        for root, _, files in os.walk(mask_dir):
            for f in files:
                stem = os.path.splitext(f)[0]
                mask_map[stem] = os.path.join(root, f)

        # Match Images and Masks
        for root, _, files in os.walk(img_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_img_exts:
                    img_stem = os.path.splitext(f)[0]
                    expected_mask_stem = img_stem + self.mask_suffix
                    
                    if expected_mask_stem in mask_map:
                        self.samples.append({
                            'image': os.path.join(root, f),
                            'label': mask_map[expected_mask_stem],
                            'case_name': img_stem
                        })
        
        if len(self.samples) == 0:
            raise ValueError(f"No matching pairs found for split '{split}'. Check paths and suffixes.")

        # Canonical deterministic ordering
        self.samples.sort(key=lambda s: os.path.normpath(s['image']).replace('\\', '/'))
        print(f"[ConfigurableDataset] Loaded {split.upper()}: {len(self.samples)} samples from {img_dir}")

        # 4. Set Transforms
        if transform:
            self.transforms = transform
        else:
            train_t, val_t = get_transformations(image_size)
            self.transforms = train_t if split == 'train' else val_t

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['label'], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        # For DeepCrack: pad to square first to preserve aspect ratio.
        # For Crack500: direct resize 640x360 -> 448x448 (no black padding borders).
        if not self.is_crack500:
            image, mask = pad_to_square(image, mask, self.image_size)

        # Apply transforms (Augmentation + Normalize + Tensor)
        augmented = self.transforms(image=image, mask=mask)
        image_out = augmented['image']
        label_out = augmented['mask']
        if label_out.dtype != torch.long:
            label_out = label_out.long()

        return {
            'image': image_out,
            'label': label_out,
            'case_name': sample['case_name'],
        }


def get_dataset_from_config(config_path, split='train', image_size=448):
    """Helper to create dataset directly from yaml path"""
    return ConfigurableMedicalDataset(config_path, split=split, image_size=image_size)
