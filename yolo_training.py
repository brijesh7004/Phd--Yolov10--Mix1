import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from yolov10_model import DetectionModel
import argparse
import os

from credentials import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, DATA_DIR
from yolo_datasets import YOLODataset, transform, collate_fn

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=80, anchors=None, strides=[8, 16, 32], reg_max=16, box_weight=7.5, cls_weight=0.5, dfl_weight=1.5):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        
        # Use anchor points if provided
        self.use_anchors = anchors is not None
        if self.use_anchors:
            self.register_buffer('anchors', torch.tensor(anchors).float())
        
        # BCE for classification
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, preds, targets):
        """
        Args:
            preds: List of predictions from different detection layers
                  Each prediction shape: [batch, num_anchors, num_classes + 4*reg_max]
            targets: Tensor of shape [num_targets, 6] (image_idx, class_id, x, y, w, h)
        Returns:
            box_loss: Bounding box regression loss
            cls_loss: Classification loss
            dfl_loss: Distribution Focal Loss for box regression
            total_loss: Combined weighted loss
        """
        device = preds[0].device
        num_targets = targets.shape[0] if len(targets) > 0 else 0
        
        # Initialize losses
        box_loss = torch.zeros(1, device=device)
        cls_loss = torch.zeros(1, device=device)
        dfl_loss = torch.zeros(1, device=device)
        
        if num_targets == 0:
            return box_loss * self.box_weight, cls_loss * self.cls_weight, dfl_loss * self.dfl_weight, (box_loss + cls_loss + dfl_loss)
        
        # Process each detection layer
        for i, pred in enumerate(preds):
            stride = self.strides[i]
            batch_size, num_anchors, _ = pred.shape
            pred = pred.view(batch_size, num_anchors, self.num_classes + 4 * self.reg_max)
            
            # Decode predictions
            pred_cls = pred[..., :self.num_classes].sigmoid()
            pred_box = pred[..., self.num_classes:].view(batch_size, num_anchors, 4, self.reg_max).softmax(-1)
            
            # Get targets for this layer
            layer_targets = targets[targets[:, 0] == i]  # Filter by image_idx
            if layer_targets.shape[0] == 0:
                continue
                
            # Convert targets to grid coordinates
            grid_size = pred.shape[2]
            target_boxes = layer_targets[:, 2:6] * grid_size  # Scale to grid
            target_classes = layer_targets[:, 1].long()
            
            # Assign targets to anchors
            if self.use_anchors:
                # Calculate anchor assignment
                anchor_indices = self.assign_anchors(target_boxes)
            else:
                # Center assignment (anchor-free)
                grid_coords = target_boxes[:, :2].long()
                anchor_indices = grid_coords[:, 1] * grid_size + grid_coords[:, 0]
            
            # Classification loss
            target_cls = torch.zeros_like(pred_cls)
            target_cls[range(len(anchor_indices)), anchor_indices, target_classes] = 1.0
            cls_loss += self.bce(pred_cls, target_cls).mean()
            
            # Box regression loss
            if self.use_anchors:
                # Anchor-based box regression
                target_boxes = self.encode_boxes(target_boxes, anchor_indices, stride)
            else:
                # Anchor-free box regression
                target_boxes = target_boxes / stride
                
            # Distribution Focal Loss for box regression
            pred_box = pred_box[range(len(anchor_indices)), anchor_indices]
            dfl_loss += self.distribution_focal_loss(pred_box, target_boxes)
            
            # CIoU Loss
            box_loss += self.compute_iou_loss(pred_box, target_boxes)
        
        # Normalize losses
        num_layers = len(preds)
        box_loss /= num_layers
        cls_loss /= num_layers
        dfl_loss /= num_layers
        
        total_loss = (self.box_weight * box_loss + 
                     self.cls_weight * cls_loss + 
                     self.dfl_weight * dfl_loss)
        
        return box_loss, cls_loss, dfl_loss, total_loss
    
    def assign_anchors(self, boxes):
        """Assign targets to anchors based on IoU"""
        if not self.use_anchors:
            return None
            
        # Calculate IoU between boxes and anchors
        boxes_wh = boxes[:, 2:]
        ious = self.bbox_iou(boxes_wh.unsqueeze(1), self.anchors.unsqueeze(0))
        best_anchor = ious.argmax(1)
        return best_anchor
    
    def encode_boxes(self, boxes, anchor_indices, stride):
        """Encode box coordinates relative to anchors"""
        anchors = self.anchors[anchor_indices] * stride
        xy = (boxes[:, :2] - anchors[:, :2]) / anchors[:, 2:]
        wh = torch.log(boxes[:, 2:] / anchors[:, 2:])
        return torch.cat([xy, wh], dim=1)
    
    def distribution_focal_loss(self, pred, target):
        """Distribution Focal Loss for box regression"""
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = 1 - weight_left
        
        loss_left = F.cross_entropy(pred, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(pred, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean()
    
    def compute_iou_loss(self, pred, target):
        """Calculate CIoU loss"""
        pred_box = self.dfl_decode(pred)
        target_box = target
        
        # Calculate CIoU
        iou = self.bbox_iou(pred_box, target_box, CIoU=True)
        return (1.0 - iou).mean()
    
    def dfl_decode(self, pred):
        """Decode DFL predictions to box coordinates"""
        project = torch.linspace(0, self.reg_max - 1, self.reg_max, device=pred.device)
        pred = pred @ project
        return pred.view(-1, 4)
    
    @staticmethod
    def bbox_iou(box1, box2, CIoU=False):
        """
        Calculate IoU between two sets of boxes
        Args:
            box1: Tensor of shape [N, 4] (x1, y1, x2, y2)
            box2: Tensor of shape [M, 4] (x1, y1, x2, y2)
            CIoU: Whether to use Complete IoU
        Returns:
            iou: Tensor of shape [N, M]
        """
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=1)
        
        # Intersection area
        inter_x1 = torch.max(b1_x1, b2_x1.T)
        inter_y1 = torch.max(b1_y1, b2_y1.T)
        inter_x2 = torch.min(b1_x2, b2_x2.T)
        inter_y2 = torch.min(b1_y2, b2_y2.T)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area.T - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-7)
        
        if CIoU:
            # Enclose area
            c_x1 = torch.min(b1_x1, b2_x1.T)
            c_y1 = torch.min(b1_y1, b2_y1.T)
            c_x2 = torch.max(b1_x2, b2_x2.T)
            c_y2 = torch.max(b1_y2, b2_y2.T)
            c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
            
            # Center distance
            b1_cx = (b1_x1 + b1_x2) / 2
            b1_cy = (b1_y1 + b1_y2) / 2
            b2_cx = (b2_x1 + b2_x2) / 2
            b2_cy = (b2_y1 + b2_y2) / 2
            center_dist = (b1_cx - b2_cx.T).pow(2) + (b1_cy - b2_cy.T).pow(2)
            
            # Diagonal distance
            diagonal_dist = (c_x2 - c_x1).pow(2) + (c_y2 - c_y1).pow(2)
            
            # Aspect ratio
            v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan((b2_x2 - b2_x1) / (b2_y2 - b2_y1 + 1e-7)).T - 
                torch.atan((b1_x2 - b1_x1) / (b1_y2 - b1_y1 + 1e-7)), 2)
            
            with torch.no_grad():
                alpha = v / (1 - iou + v + 1e-7)
            
            # CIoU
            iou = iou - (center_dist / (diagonal_dist + 1e-7) + alpha * v)
        
        return iou

def train(args):
    # Initialize model
    model = DetectionModel(variant=args.variant)
    
    # Load pretrained weights if specified
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
        print(f"Loaded pretrained weights from {args.pretrained}")
    
    # Create datasets
    train_dataset = YOLODataset(
        images_dir=os.path.join(args.train_data, 'images', 'train'),
        labels_dir=os.path.join(args.train_data, 'labels', 'train'),
        img_size=args.img_size, transform=transform
    )
    val_dataset = YOLODataset(
        images_dir=os.path.join(args.val_data, 'images', 'val'),
        labels_dir=os.path.join(args.val_data, 'labels', 'val'),
        img_size=args.img_size, transform=transform
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    
    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[args.epochs//2, args.epochs*3//4], gamma=0.1)
    
    # Loss function (you'll need to implement this according to YOLOv10's loss)
    criterion = YOLOLoss(num_classes=80, anchors=None, strides=[8, 16, 32], reg_max=16)
    
    # Training loop
    best_val_loss = float('inf')
    model.to(args.device)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(args.device)
            targets = [t.to(args.device) for t in targets]
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            box_loss, cls_loss, dfl_loss, total_loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if batch_idx % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(args.device)
                targets = [t.to(args.device) for t in targets]
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pth'))
        
        # Save checkpoint
        if (epoch+1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'checkpoint_{epoch+1}.pth'))
        
        scheduler.step()
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='m', help='Model variant')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Root directory of dataset')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['EPOCHS'], help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['BATCH_SIZE'], help='Batch size')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['LEARNING_RATE'], help='Learning rate')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--workers', type=int, default=MODEL_CONFIG['NUM_WORKERS'], help='Number of data loader workers')
    parser.add_argument('--log_interval', type=int, default=TRAIN_CONFIG['LOG_INTERVAL'], help='Log interval in batches')
    parser.add_argument('--save_interval', type=int, default=TRAIN_CONFIG['SAVE_INTERVAL'], help='Save interval in epochs')
    parser.add_argument('--output_dir', type=str, default='weights', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)