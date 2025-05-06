"""
YOLOv10 Dataset Class
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A

class YOLODataset(Dataset):
    """Dataset class for YOLOv10."""
    
    def __init__(self, img_dir, label_dir, img_size=640, augment=False):
        """Initialize dataset."""
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Get image files
        self.img_files = sorted([
            f for f in self.img_dir.glob('*')
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        
        # Verify label files exist
        self.label_files = []
        for img_file in self.img_files:
            label_file = self.label_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                print(f"Warning: No label file for {img_file}")
                continue
            self.label_files.append(label_file)
        
        # Remove images without labels
        self.img_files = [
            img for img, lbl in zip(self.img_files, self.label_files)
            if lbl.exists()
        ]
        
        # Setup augmentations
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))
        else:
            self.transform = None
    
    def __len__(self):
        """Return dataset length."""
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """Get dataset item."""
        # Load image
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = self.label_files[idx]
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f.readlines():
                    class_id, *box = map(float, line.strip().split())
                    labels.append([class_id, *box])
        labels = np.array(labels)
        
        # Get original dimensions
        height, width = img.shape[:2]
        
        # Apply augmentations if enabled
        if self.transform and len(labels):
            transformed = self.transform(
                image=img,
                bboxes=labels[:, 1:],
                class_labels=labels[:, 0]
            )
            img = transformed['image']
            if len(transformed['bboxes']):
                labels = np.column_stack([
                    transformed['class_labels'],
                    transformed['bboxes']
                ])
        
        # Resize and pad image
        r = self.img_size / max(height, width)
        if r != 1:
            interp = cv2.INTER_LINEAR
            img = cv2.resize(img, (int(width * r), int(height * r)), interpolation=interp)
        
        new_height, new_width = img.shape[:2]
        dw, dh = self.img_size - new_width, self.img_size - new_height
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        
        # Add padding
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img = img / 255.0  # Normalize
        
        # Convert labels to absolute coordinates
        if len(labels):
            # Convert normalized xywh to pixel xyxy format
            labels[:, 1:] *= np.array([width, height, width, height])
            labels[:, 1] = labels[:, 1] * r + left
            labels[:, 2] = labels[:, 2] * r + top
            labels[:, 3] = labels[:, 3] * r
            labels[:, 4] = labels[:, 4] * r
        
        # Create target dictionary
        target = {
            'boxes': torch.from_numpy(labels[:, 1:]).float() if len(labels) else torch.zeros((0, 4)),
            'labels': torch.from_numpy(labels[:, 0]).long() if len(labels) else torch.zeros(0),
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([height, width])
        }
        
        return img, target
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader."""
        images = []
        targets = []
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        images = torch.stack(images, 0)
        return images, targets

def create_dataloader(dataset, batch_size=16, num_workers=4, shuffle=True):
    """Create a DataLoader for the dataset."""
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
