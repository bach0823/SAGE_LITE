import os
import random
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
import cv2
from sklearn.model_selection import train_test_split
import logging
Image.MAX_IMAGE_PIXELS = None 


def pad_to_square(image: np.ndarray, mask: np.ndarray, min_size: int = 448):
    """
    Dynamic Pad-to-Square based on actual image dimensions:
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


def get_transformations(img_size, crop_mode='random'):
    """
    Get data augmentation transformations for training and validation
    """
    if crop_mode == 'random':
        base_crop = [
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT_101, fill_mask=0),
            A.RandomCrop(width=img_size, height=img_size)
        ]
        val_crop = [
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT_101, fill_mask=0),
        ]
    else:
        # Dynamic Pad-to-Square is performed in __getitem__ before transforms.
        # Resize then scales the square image to (img_size, img_size) with nearest mask interpolation.
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
        A.CLAHE(clip_limit=2.0, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    
    train_transform = A.Compose(base_crop + aug_list)
    crop_transform = A.Compose(base_crop)
    aug_transform = A.Compose(aug_list)
    
    val_transforms = A.Compose(val_crop + [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transforms, crop_transform, aug_transform

class UniversalMedicalDataset(Dataset):
    """
    A generic dataset loader for medical segmentation tasks.
    Assumes a flat directory structure or simple subfolders.
    Automatically splits data into train/val/test if not pre-split.
    """
    def __init__(self, root_dir, split='train', transform=None, 
                 img_ext='.png', mask_suffix='_mask', 
                 split_ratios=(0.7, 0.15, 0.15), seed=42, image_size=512):
        """
        Args:
            root_dir: Path to the dataset root.
            split: 'train', 'val', or 'test'.
            transform: Albumentations transforms (optional, will use default if None).
            img_ext: Extension of images (e.g., '.png', '.jpg').
            mask_suffix: Suffix for mask files (e.g., '_mask', '_seg').
            split_ratios: Tuple of (train, val, test) ratios.
            seed: Random seed for reproducibility.
            image_size: Target image size for transformations.
        """
        self.root_dir = root_dir
        self.split = split
        self.mask_suffix = mask_suffix
        self.img_ext = img_ext
        self.image_size = image_size
        
        # 1. Collect all image files
        self.all_images = []
        self.all_masks = []
        self.case_names = []
        
        # Walk through directory to find images
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(img_ext) and mask_suffix not in file:
                    img_path = os.path.join(root, file)
                    
                    # Construct expected mask path
                    # Assumes mask is in the same folder or a parallel 'masks' folder
                    # Strategy 1: Same folder, suffix check
                    mask_name = file.replace(img_ext, mask_suffix + img_ext)
                    mask_path = os.path.join(root, mask_name)
                    
                    # Strategy 2: Check if mask exists
                    if os.path.exists(mask_path):
                        self.all_images.append(img_path)
                        self.all_masks.append(mask_path)
                        self.case_names.append(os.path.splitext(file)[0])
                    else:
                        # Strategy 3: Try 'masks' subfolder if images are in 'images'
                        # This is a simple heuristic; can be expanded
                        pass

        if len(self.all_images) == 0:
            print(f"Warning: No images found in {root_dir} with extension {img_ext} and mask suffix {mask_suffix}")
            
        # 2. Split the dataset
        # Use case_names to ensure consistent splitting
        indices = list(range(len(self.all_images)))
        
        # First split: Train vs (Val + Test)
        train_idx, temp_idx = train_test_split(
            indices, train_size=split_ratios[0], random_state=seed, shuffle=True
        )
        
        # Second split: Val vs Test
        # Normalize ratios for the remaining part
        remaining_ratio = split_ratios[1] + split_ratios[2]
        val_ratio_norm = split_ratios[1] / remaining_ratio
        
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_ratio_norm, random_state=seed, shuffle=True
        )
        
        # Select indices based on split
        if split == 'train':
            self.indices = train_idx
        elif split == 'val':
            self.indices = val_idx
        else:
            self.indices = test_idx
            
        print(f"[UniversalDataset] {split.upper()}: {len(self.indices)} samples from {root_dir}")
        
        # 3. Set Transforms
        if transform:
            self.transform = transform
        else:
            train_t, val_t, _, _ = get_transformations(image_size)
            self.transform = train_t if split == 'train' else val_t

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        img_path = self.all_images[real_idx]
        mask_path = self.all_masks[real_idx]
        
        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
        
        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            label = augmented['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(mask).long()
            
        # Ensure label is long
        if label.dtype != torch.long:
            label = label.long()
            
        return {
            'image': image,
            'label': label,
            'case_name': self.case_names[real_idx]
        }

class ConfigurableMedicalDataset(Dataset):
    """
    Dataset loader that reads from a YAML configuration file.
    Supports pre-split datasets (train/val/test in separate folders).
    """
    def __init__(self, config_path_or_dict, split='train', image_size=512, transform=None):
        """
        Args:
            config_path_or_dict: Path to .yaml file OR a dictionary containing config.
            split: 'train', 'val', or 'test'.
            image_size: Target size for resizing.
            transform: Optional custom transform. If None, uses default based on split.
        """
        self.split = split
        self.image_size = image_size
        
        # 1. Load Configuration
        if isinstance(config_path_or_dict, str):
            with open(config_path_or_dict, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_path_or_dict

        root_dir = self.config.get('root_dir', '')
        
        # Branch logic: DeepCrack vs Crack500 based on config explicitly, or auto-detect
        self.is_crack500 = 'Crack500' in root_dir
        self.use_smart_filter = self.config.get('smart_filter', self.is_crack500)
        
        # 2. Get paths for the specific split
        if split not in self.config:
            # Fallback: if 'test' is requested but not in config, use 'val' or raise error
            # For now, strict mode
            raise ValueError(f"Split '{split}' not found in configuration file.")
            
        split_config = self.config[split]
        
        # Handle relative or absolute paths
        img_dir = split_config['images']
        mask_dir = split_config['masks']
        
        if root_dir and not os.path.isabs(img_dir):
            img_dir = os.path.join(root_dir, img_dir)
        if root_dir and not os.path.isabs(mask_dir):
            mask_dir = os.path.join(root_dir, mask_dir)
            
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Image dir '{img_dir}' or Mask dir '{mask_dir}' does not exist.")

        # 3. Settings
        self.mask_suffix = self.config.get('mask_suffix', '')
        # Auto-detect common medical image formats (no need to define in YAML)
        valid_img_exts = set(['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.dcm', '.nii', '.nii.gz'])
        
        # 4. Match Images and Masks
        self.samples = []
        
        # Index all masks by stem (filename without extension)
        mask_map = {}
        for root, _, files in os.walk(mask_dir):
            for f in files:
                stem = os.path.splitext(f)[0]
                mask_map[stem] = os.path.join(root, f)
        
        # Iterate images and find matching masks
        for root, _, files in os.walk(img_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_img_exts:
                    img_stem = os.path.splitext(f)[0]
                    
                    # Construct expected mask stem
                    # e.g. Image: "case1", Suffix: "_seg" -> Mask: "case1_seg"
                    expected_mask_stem = img_stem + self.mask_suffix
                    
                    if expected_mask_stem in mask_map:
                        self.samples.append({
                            'image': os.path.join(root, f),
                            'label': mask_map[expected_mask_stem],
                            'case_name': img_stem
                        })
        
        if len(self.samples) == 0:
            raise ValueError(f"No matching pairs found for split '{split}'. Check paths and suffixes.")

        # Canonical deterministic ordering across all platforms/filesystems
        self.samples.sort(key=lambda s: os.path.normpath(s['image']).replace('\\', '/'))

        print(f"[ConfigurableDataset] Loaded {split.upper()}: {len(self.samples)} samples from {img_dir}")

        # 5. Set Transforms
        self.crop_mode = self.config.get('crop_mode', 'random')
        if transform:
            self.transforms = transform
            self.crop_transform = None
            self.aug_transform = None
        else:
            # Default transforms
            train_t, val_t, crop_t, aug_t = get_transformations(image_size, crop_mode=self.crop_mode)
            self.transforms = train_t if split == 'train' else val_t
            self.crop_transform = crop_t if split == 'train' else None
            self.aug_transform = aug_t if split == 'train' else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import random

        # ── Smart-filter path (Crack500 train) ────────────────────────────────
        # Accepts only crops with fg_pixels >= 20 (crack present).
        # Retries up to 20 crops per source; resamples source up to 10 times.
        # Val/Test bypass this filter.
        if self.split == 'train' and self.crop_transform and self.use_smart_filter:
            MAX_RESAMPLE = 10
            for source_try in range(MAX_RESAMPLE):
                sample = self.samples[idx]

                image = cv2.imread(sample['image'])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask  = cv2.imread(sample['label'], cv2.IMREAD_GRAYSCALE)
                mask  = (mask > 0).astype(np.uint8)

                success = False
                for _ in range(20):
                    cropped   = self.crop_transform(image=image, mask=mask)
                    fg_pixels = (cropped['mask'] > 0).sum()
                    if fg_pixels >= 20:
                        success = True
                        break

                if not success:
                    idx = random.randint(0, len(self.samples) - 1)
                    continue

                augmented = self.aug_transform(image=cropped['image'], mask=cropped['mask'])
                image_out = augmented['image']
                label_out = augmented['mask']
                if label_out.dtype != torch.long:
                    label_out = label_out.long()
                return {
                    'image':     image_out,
                    'label':     label_out,
                    'case_name': sample['case_name'],
                }

            raise RuntimeError(
                f"[Crack500] FAIL-FAST: Exceeded {MAX_RESAMPLE} source resampling attempts. "
                f"All sampled sources may lack sufficient foreground (fg_pixels < 20 in all 20 crop attempts)."
            )

        # ── Standard path (val / test / DeepCrack / no smart_filter) ─────────
        sample = self.samples[idx]
        image  = cv2.imread(sample['image'])
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask   = cv2.imread(sample['label'], cv2.IMREAD_GRAYSCALE)
        # Binarize mask to {0, 1} before applying transforms
        mask   = (mask > 0).astype(np.uint8)

        # Dynamic Pad-to-Square for resize mode (DeepCrack protocol):
        # target = max(H, W, img_size), preserve aspect ratio before resizing to 448x448
        if getattr(self, 'crop_mode', 'random') == 'resize':
            image, mask = pad_to_square(image, mask, self.image_size)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image_out = augmented['image']
            label_out = augmented['mask']
        else:
            image_out = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            label_out = torch.from_numpy(mask).long()

        if label_out.dtype != torch.long:
            label_out = label_out.long()
        return {
            'image':     image_out,
            'label':     label_out,
            'case_name': sample['case_name'],
        }


def get_dataset_from_config(config_path, split='train', image_size=512):
    """Helper to create dataset directly from yaml path"""
    return ConfigurableMedicalDataset(config_path, split=split, image_size=image_size)








