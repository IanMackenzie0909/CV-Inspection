import torch
from torch.utils.data import Dataset
import numpy as np
import random
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF


class JointAugmentation:
    """
    Joint augmentation for image and mask pairs.
    Geometric transforms are applied identically to both.
    Color transforms are applied only to the image.
    """
    def __init__(self,
                 horizontal_flip=True,
                 vertical_flip=False,
                 rotation_degrees=15,
                 color_jitter=True,
                 brightness=0.2,
                 contrast=0.2,
                 saturation=0.1,
                 hue=0.05):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, mask):
        """
        Apply augmentation to image and mask.

        Args:
            image: PIL Image (RGB)
            mask: PIL Image (L/grayscale)

        Returns:
            Augmented (image, mask) tuple
        """
        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if self.vertical_flip and random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)

        # Color jitter (only for image, not mask)
        if self.color_jitter:
            # Random brightness
            if self.brightness > 0:
                brightness_factor = 1 + random.uniform(-self.brightness, self.brightness)
                image = TF.adjust_brightness(image, brightness_factor)

            # Random contrast
            if self.contrast > 0:
                contrast_factor = 1 + random.uniform(-self.contrast, self.contrast)
                image = TF.adjust_contrast(image, contrast_factor)

            # Random saturation
            if self.saturation > 0:
                saturation_factor = 1 + random.uniform(-self.saturation, self.saturation)
                image = TF.adjust_saturation(image, saturation_factor)

            # Random hue
            if self.hue > 0:
                hue_factor = random.uniform(-self.hue, self.hue)
                image = TF.adjust_hue(image, hue_factor)

        return image, mask


class SAMDataset(Dataset):
    """
    SAMDataset is a simple custom dataset class for images and their corresponding masks.
    Supports both JPG images with PNG masks (original format) and
    PNG images with _mask.png masks (tongue dataset format).
    """
    def __init__(self, root_dir, transform=None, max_bbox_shift=10, augmentation=None,
                 bbox_expansion_prob=0.3, max_bbox_expansion=0.5, full_image_bbox_prob=0.1):
        """
        Args:
            root_dir (string): Directory containing images and masks.
            transform (tuple, optional): A tuple of two optional transforms to be applied
                on an image and its mask respectively.
            bbox_shift (int, optional): Add random perturbation in the range [-bbox_shift, bbox_shift]
                to the bounding box coordinates.
            augmentation (JointAugmentation, optional): Joint augmentation for image-mask pairs.
            bbox_expansion_prob (float): Probability of expanding BBox beyond tight bounds.
            max_bbox_expansion (float): Maximum expansion ratio (0.5 = expand by up to 50% on each side).
            full_image_bbox_prob (float): Probability of using full image as BBox.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_bbox_shift = max_bbox_shift
        self.augmentation = augmentation
        self.bbox_expansion_prob = bbox_expansion_prob
        self.max_bbox_expansion = max_bbox_expansion
        self.full_image_bbox_prob = full_image_bbox_prob

        # Get all image files (both .jpg and .png)
        all_jpgs = list(self.root_dir.rglob('*.jpg'))
        all_pngs = list(self.root_dir.rglob('*.png'))

        # Filter out mask files from PNG list (files ending with _mask.png)
        all_pngs = [p for p in all_pngs if not p.stem.endswith('_mask')]

        self.img_list = []
        self.mask_list = []

        # Process JPG files (original format: image.jpg -> image.png)
        for img_path in all_jpgs:
            mask_path = img_path.with_suffix('.png')
            if mask_path.exists():
                self.img_list.append(img_path)
                self.mask_list.append(mask_path)
            else:
                print(f"Warning: {img_path} doesn't have a corresponding mask!")

        # Process PNG files (tongue format: image.png -> image_mask.png)
        for img_path in all_pngs:
            mask_path = img_path.parent / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                self.img_list.append(img_path)
                self.mask_list.append(mask_path)
            else:
                print(f"Warning: {img_path} doesn't have a corresponding mask!")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Fetch an image-mask pair by index.

        Args:
        - idx (int): Index of the desired sample.

        Returns:
        - tuple: An (image, mask) pair.
        """
        img_name = self.img_list[idx]
        mask_name = self.mask_list[idx]

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")

        # CRITICAL FIX: Binarize mask to 0 and 255
        # Some datasets have masks with values like 0/1 or 0/38 instead of 0/255
        # This ensures mask values are properly scaled after ToTensor()
        import numpy as np
        mask_np = np.array(mask)
        mask_np = (mask_np > 0).astype(np.uint8) * 255  # Binary: 0 or 255
        mask = Image.fromarray(mask_np, mode='L')

        # Apply joint augmentation (before transforms) - only for training
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)

        # Apply transformations if any
        if self.transform:
            image = self.transform[0](image)
            mask = self.transform[1](mask)

        x_min, y_min, x_max, y_max = self.compute_bbox(mask.squeeze(0))

        # Add random perturbation for data augmentation
        c, h, w = mask.shape  # Note: after ToTensor, shape is [C, H, W]
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # BBox augmentation strategy:
        # 1. With full_image_bbox_prob, use full image as BBox
        # 2. With bbox_expansion_prob, expand BBox beyond tight bounds
        # 3. Otherwise, add small random perturbation

        if random.random() < self.full_image_bbox_prob:
            # Use full image as BBox - teaches model to handle automatic detection
            x_min, y_min, x_max, y_max = 0, 0, w, h
        elif random.random() < self.bbox_expansion_prob:
            # Expand BBox by random amount on each side
            expand_x = random.uniform(0, self.max_bbox_expansion) * bbox_width
            expand_y = random.uniform(0, self.max_bbox_expansion) * bbox_height
            x_min = max(0, int(x_min - expand_x))
            x_max = min(w, int(x_max + expand_x))
            y_min = max(0, int(y_min - expand_y))
            y_max = min(h, int(y_max + expand_y))
        else:
            # Original: small random shift
            noise_w = torch.clamp(torch.randn(1) * bbox_width * 0.1, min=-self.max_bbox_shift, max=self.max_bbox_shift).round().int().item()
            noise_h = torch.clamp(torch.randn(1) * bbox_height * 0.1, min=-self.max_bbox_shift, max=self.max_bbox_shift).round().int().item()
            x_min = max(0, x_min + noise_w)
            x_max = min(w, x_max + noise_w)
            y_min = max(0, y_min + noise_h)
            y_max = min(h, y_max + noise_h)

        bboxes = torch.tensor([x_min, y_min, x_max, y_max])

        return image, mask, bboxes
    
    def compute_bbox(self, mask_tensor):
        """
        Compute the bounding box of the white region in a binary mask tensor.
        
        Args:
            mask_tensor (tensor): A binary mask tensor. Assumes white as 1 and black as 0.
            
        Returns:
            tensor: A tensor containing coordinates (x_min, y_min, x_max, y_max) of the bbox.
        """
        # Assuming input is a PIL Image. Convert to tensor and squeeze if necessary
        if len(mask_tensor.shape) > 2:
            mask_tensor = mask_tensor.squeeze(0)

        # Detect which rows and columns have white pixels
        rows_any_white = torch.any(mask_tensor == 1, dim=1)
        cols_any_white = torch.any(mask_tensor == 1, dim=0)

        # Get the min and max row and column indices with white pixels
        rows_white = torch.where(rows_any_white)[0]
        cols_white = torch.where(cols_any_white)[0]

        if rows_white.nelement() == 0 or cols_white.nelement() == 0:
            # No white pixels, return zeros
            return torch.tensor([0, 0, 0, 0])

        y_min, y_max = rows_white[0].item(), rows_white[-1].item()
        x_min, x_max = cols_white[0].item(), cols_white[-1].item()

        return x_min, y_min, x_max, y_max