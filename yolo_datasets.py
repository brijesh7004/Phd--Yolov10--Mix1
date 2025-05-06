import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    # def __getitem__(self, idx):
    #     img_path = os.path.join(self.images_dir, self.image_files[idx])
    #     label_path = os.path.join(self.labels_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')
        
    #     # Load image
    #     img = Image.open(img_path).convert('RGB')
    #     width, height = img.size
        
    #     # Apply transformations if specified
    #     if self.transform:
    #         img = self.transform(img)
    #     else:
    #         img = transforms.ToTensor()(img)
        
    #     # Load labels
    #     targets = []
    #     if os.path.exists(label_path):
    #         with open(label_path, 'r') as f:
    #             for line in f.readlines():
    #                 line = line.strip().split()
    #                 if len(line) == 5:  # class_id, x_center, y_center, width, height (normalized)
    #                     class_id = int(line[0])
    #                     x_center = float(line[1])
    #                     y_center = float(line[2])
    #                     box_width = float(line[3])
    #                     box_height = float(line[4])
                        
    #                     # Convert to absolute coordinates
    #                     x1 = (x_center - box_width/2) * width
    #                     y1 = (y_center - box_height/2) * height
    #                     x2 = (x_center + box_width/2) * width
    #                     y2 = (y_center + box_height/2) * height
                        
    #                     targets.append([class_id, x1, y1, x2, y2])
        
    #     # Convert targets to tensor
    #     targets = torch.tensor(targets, dtype=torch.float32) if len(targets) > 0 else torch.zeros((0, 5))
        
    #     return img, targets
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        # Load and normalize labels
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip().split()
                    if len(line) == 5:
                        class_id = int(line[0])
                        x_center = float(line[1])  # Already normalized [0,1]
                        y_center = float(line[2])
                        box_width = float(line[3])
                        box_height = float(line[4])
                        targets.append([class_id, x_center, y_center, box_width, box_height])
        
        targets = torch.tensor(targets, dtype=torch.float32) if len(targets) > 0 else torch.zeros((0, 5))
        return img, targets

# Define transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# def collate_fn(batch):
#     images, targets = zip(*batch)
#     images = torch.stack(images, 0)  # Stack images in the batch
    
#     # Handle targets (labels) which may have different lengths
#     targets = [target for target in targets if len(target) > 0]
#     return images, targets
# def collate_fn(batch):
#     """Custom collate function to handle variable numbers of bounding boxes"""
#     images, targets = list(zip(*batch))
    
#     # Pad images to same size in batch
#     images = torch.stack(images, 0)
    
#     # Create list of targets with image index
#     new_targets = []
#     for i, boxes in enumerate(targets):
#         if boxes.shape[0] > 0:
#             img_idx = torch.full((boxes.shape[0], 1), i)
#             boxes_with_idx = torch.cat([img_idx, boxes], dim=1)
#             new_targets.append(boxes_with_idx)
    
#     # Concatenate all targets
#     targets = torch.cat(new_targets, 0) if len(new_targets) > 0 else torch.zeros((0, 6))
    
#     return images, targets
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    
    # Add image index to targets
    new_targets = []
    for i, boxes in enumerate(targets):
        if boxes.shape[0] > 0:
            img_idx = torch.full((boxes.shape[0], 1), i)
            boxes_with_idx = torch.cat([img_idx, boxes], dim=1)
            new_targets.append(boxes_with_idx)
    
    targets = torch.cat(new_targets, 0) if len(new_targets) > 0 else torch.zeros((0, 6))
    return images, targets