import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from yolov10_model import DetectionModel  # Assuming your model is in model.py
import os

from credentials import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, DATA_DIR
from yolo_datasets import YOLODataset, transform, collate_fn
num_classes = MODEL_CONFIG['NUM_CLASSES']


def bbox_iou(box1, box2, x1y1x2y2=True, CIoU=False, eps=1e-7):
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        box1 (Tensor): Predicted boxes [N, 4]
        box2 (Tensor): Target boxes [N, 4]
        x1y1x2y2 (bool): If True, boxes are [x1,y1,x2,y2]. Else [cx,cy,w,h]
        CIoU (bool): If True, use Complete IoU loss
        eps (float): Small value to avoid division by zero
    """
    # Get box coordinates
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    else:
        # Convert from center-width-height to xyxy
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2]/2, box1[:, 0] + box1[:, 2]/2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3]/2, box1[:, 1] + box1[:, 3]/2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2]/2, box2[:, 0] + box2[:, 2]/2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3]/2, box2[:, 1] + box2[:, 3]/2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    
    if CIoU:
        # Implement Complete IoU (https://arxiv.org/abs/1911.08287)
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        
        # Diagonal distance of convex shape
        c2 = cw**2 + ch**2 + eps
        
        # Center distance
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 + 
               (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2) / 4
        
        # Aspect ratio
        v = (4 / math.pi**2) * torch.pow(torch.atan(w2/h2) - torch.atan(w1/h1), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        
        return iou - (rho2 / c2 + v * alpha)
    
    return iou


def distribution_focal_loss(pred, target):
    """
    Calculates the Distribution Focal Loss (DFL).

    Args:
        pred (torch.Tensor): Predicted DFL distributions (e.g., after reshaping).
                             Shape: (..., 4, reg_max)
        target (torch.Tensor): Target DFL distributions (e.g., indices and weights).
                               Shape: (..., 4) for indices, (..., 4) for weights (or similar)

    Returns:
        torch.Tensor: The DFL loss.
    """
    # Implement your DFL loss logic here.
    # This involves applying softmax to 'pred' and then calculating
    # the weighted cross-entropy with 'target'.
    # This is a placeholder.
    print("Warning: Using placeholder distribution_focal_loss function.")
    # Dummy implementation for demonstration
    return torch.tensor(0.1) # Example dummy return

def decode_bboxes(initial_box_preds, dfl_preds, anchors, stride, reg_max):
    """
    Decodes the predicted bounding box offsets from initial predictions and DFL.

    Args:
        initial_box_preds (torch.Tensor): Initial box predictions (e.g., cx, cy, w, h offsets).
                                          Shape: (..., 4)
        dfl_preds (torch.Tensor): DFL predictions. Shape: (..., 4 * reg_max)
        anchors (torch.Tensor): Anchor box coordinates or similar reference points.
                                Shape: (..., 4) or (num_locations, 4)
        stride (int): The downsampling factor of the feature map.
        reg_max (int): The maximum value for DFL bins.

    Returns:
        torch.Tensor: Decoded bounding box coordinates (e.g., x1, y1, x2, y2).
                      Shape: (..., 4)
    """
    # Implement your bounding box decoding logic here.
    # This involves using the initial predictions and the DFL distributions
    # to refine the bounding box coordinates relative to the anchors/locations.
    # This is a placeholder.
    print("Warning: Using placeholder decode_bboxes function.")
    # Dummy implementation for demonstration
    return torch.zeros_like(initial_box_preds) # Example dummy return

# --- End of Placeholder Helper Functions ---


class YOLOLoss(nn.Module):
    def __init__(self, nc, reg_max=16, **kwargs):
        """
        Initializes the YOLO loss function.

        Args:
            nc (int): Number of classes.
            reg_max (int): Maximum value for DFL bins.
            kwargs: Additional arguments (e.g., loss weights).
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.reg_max = reg_max  # Maximum value for DFL bins

        # Loss function components
        self.bce = nn.BCEWithLogitsLoss(reduction='none') # Binary Cross Entropy for objectness and classes
        self.iou_loss = bbox_iou  # Using bbox_iou (assuming it includes CIoU) for bounding boxes
        self.dfl_loss = distribution_focal_loss # Distribution Focal Loss for bounding boxes

        # Loss weights (example values, tune as needed)
        self.obj_weight = kwargs.get('obj_weight', 1.0)
        self.cls_weight = kwargs.get('cls_weight', 0.5)
        self.box_weight = kwargs.get('box_weight', 0.02) # Box loss is sum of DFL and CIoU

        # Assuming the model outputs per location:
        # 4 initial box | 1 objectness | nc classes | 4*reg_max DFL features
        self.num_features_per_location = 4 + 1 + self.nc + 4 * self.reg_max

    def forward(self, x, targets, anchors=None, stride=None):
        """
        Calculates the YOLO loss.

        Args:
            x (torch.Tensor): Model's output tensor (batch_size, num_features, height, width).
                              num_features should be 4 + 1 + nc + 4*reg_max.
            targets (torch.Tensor): Ground truth targets (batch_size, num_targets, 5+nc).
                                    Format: (image_index, class_index, cx, cy, w, h, [class_one_hot])
            anchors (torch.Tensor): Anchor box coordinates or similar reference points.
                                    Shape: (num_locations, 4)
            stride (int or float): The downsampling factor of the feature map.

        Returns:
            tuple: A tuple containing the total loss and a dictionary of loss components.
        """
        # Ensure the model output has the expected number of features
        batch_size, num_features, height, width = x.shape
        if num_features != self.num_features_per_location:
            raise ValueError(f"Model output has {num_features} features, "
                             f"but YOLOLoss expects {self.num_features_per_location} features per location "
                             f"(4 initial box + 1 obj + {self.nc} cls + 4*{self.reg_max} DFL).")

        num_locations = height * width

        # Reshape x to (batch_size, num_locations, num_features_per_location)
        x = x.view(batch_size, num_features, -1).transpose(1, 2)

        # Slice the predictions
        # Initial box, objectness, and class predictions
        box_obj_cls_preds = x[:, :, :5 + self.nc] # (batch_size, num_locations, 5 + nc)

        # DFL features
        dfl_preds = x[:, :, 5 + self.nc:] # (batch_size, num_locations, 4 * reg_max)

        # Separate initial box, objectness, and class predictions
        initial_box_preds = box_obj_cls_preds[:, :, :4] # (batch_size, num_locations, 4)
        obj_preds = box_obj_cls_preds[:, :, 4] # (batch_size, num_locations)
        cls_preds = box_obj_cls_preds[:, :, 5:] # (batch_size, num_locations, nc)

        # --- Target Assignment ---
        # This is the most complex part. You need to determine which predictions
        # are responsible for which ground truth objects.
        # This typically involves:
        # 1. Generating anchor/location coordinates for the feature map.
        # 2. Calculating IoUs between ground truth boxes and potential predictions.
        # 3. Assigning ground truth boxes to the best matching predictions based on criteria
        #    (e.g., IoU threshold, center proximity).
        # 4. Creating target tensors for objectness, class, and bounding box regression.

        # Placeholder for target assignment outputs:
        # positive_mask: A boolean tensor indicating which locations are assigned to a ground truth.
        # obj_targets: Target objectness scores (typically 1 for positives, 0 for negatives).
        # cls_targets: Target class scores (typically one-hot encoded for positives).
        # box_targets: Target bounding box values (e.g., cx, cy, w, h offsets or DFL targets).
        # gt_bboxes: Ground truth bounding boxes corresponding to positive predictions.

        # Example Placeholder Target Assignment (Highly Simplified)
        # You will need a proper target assignment mechanism.
        positive_mask = torch.zeros_like(obj_preds, dtype=torch.bool)
        obj_targets = torch.zeros_like(obj_preds)
        cls_targets = torch.zeros_like(cls_preds)
        box_targets = {} # Dictionary for different box target types (e.g., DFL targets)
        gt_bboxes = None
        assigned_locations = None # Coordinates of assigned locations

        # --- Loss Calculation ---

        # Initialize loss components
        loss_obj = torch.tensor(0., device=x.device)
        loss_cls = torch.tensor(0., device=x.device)
        loss_box = torch.tensor(0., device=x.device) # Total box loss (DFL + CIoU)
        loss_dfl = torch.tensor(0., device=x.device)
        loss_ciou = torch.tensor(0., device=x.device)

        # Only calculate losses for positive locations
        if positive_mask.sum() > 0:
            positive_obj_preds = obj_preds[positive_mask]
            positive_cls_preds = cls_preds[positive_mask]
            positive_initial_box_preds = initial_box_preds[positive_mask]
            positive_dfl_preds = dfl_preds[positive_mask]

            # 1. Objectness Loss (for both positive and negative locations)
            # Need obj_targets for all locations after assignment
            loss_obj = self.bce(obj_preds, obj_targets).mean() # Mean over all locations

            # 2. Class Loss (only for positive locations)
            # Need cls_targets for positive locations
            loss_cls = self.bce(positive_cls_preds, cls_targets[positive_mask]).mean()

            # 3. Bounding Box Loss (DFL + CIoU) (only for positive locations)

            # Calculate DFL loss
            # Reshape DFL predictions for DFL loss function: (num_pos, 4 * reg_max) -> (num_pos, 4, reg_max)
            dfl_preds_reshaped = positive_dfl_preds.view(-1, 4, self.reg_max)
            # Need DFL targets for positive locations
            # Assuming box_targets['dfl'] contains the necessary targets for DFL loss
            loss_dfl = self.dfl_loss(dfl_preds_reshaped, box_targets['dfl'])

            # Decode bounding boxes from initial predictions and DFL for CIoU loss
            # Need assigned_locations (coordinates) and stride for decoding
            # Need gt_bboxes for positive locations
            decoded_bboxes = decode_bboxes(
                positive_initial_box_preds,
                positive_dfl_preds,
                assigned_locations, # Pass location coordinates
                stride,
                self.reg_max
            )
            loss_ciou = 1.0 - self.iou_loss(decoded_bboxes, gt_bboxes, CIoU=True).mean()

            # Total box loss
            loss_box = loss_dfl + loss_ciou

        # Total Loss
        total_loss = (self.obj_weight * loss_obj +
                      self.cls_weight * loss_cls +
                      self.box_weight * loss_box)

        # Return loss components for logging
        loss_components = {
            'obj_loss': loss_obj.item(),
            'cls_loss': loss_cls.item(),
            'box_loss': loss_box.item(),
            'dfl_loss': loss_dfl.item(),
            'ciou_loss': loss_ciou.item(),
        }

        return loss_box, loss_cls, loss_dfl, total_loss
        # return total_loss, loss_components


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils.metrics import bbox_iou

class YOLOLoss2(nn.Module):
    def __init__(self, nc=80, reg_max=16, use_dfl=True):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.box_weight = 10    #7.5
        self.cls_weight = 0.5
        self.dfl_weight = 2     #1.5
        
        # Define anchors for each prediction layer (example values, adjust based on your model)
        # Format: [width, height] for each anchor at each scale
        self.anchors = {
            0: torch.tensor([[10,13], [16,30], [33,23]]),    # P3 anchors
            1: torch.tensor([[30,61], [62,45], [59,119]]),   # P4 anchors
            2: torch.tensor([[116,90], [156,198], [373,326]]) # P5 anchors
        }
        
        # Anchor grid counts (how many anchors per position)
        self.na = {0: 3, 1: 3, 2: 3}  # 3 anchors per position for each scale

    @staticmethod
    def _wh_iou(wh1, wh2):
        """Calculate IoU between two sets of widths/heights."""
        wh1 = wh1[:, None]  # [N,1,2]
        wh2 = wh2[None]     # [1,M,2]
        inter = torch.min(wh1, wh2).prod(2)  # [N,M]
        return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)

    def forward(self, preds, targets):
        device = preds[0].device
        box_loss = torch.zeros(1, device=device)
        cls_loss = torch.zeros(1, device=device)
        dfl_loss = torch.zeros(1, device=device)

        for i, pred in enumerate(preds):
            stride = 8 * (2 ** i)  # P3:8, P4:16, P5:32
            pred = pred.permute(0, 2, 3, 1)  # [B, H, W, C]
            b, h, w, c = pred.shape
            
            # Calculate expected feature size
            expected_features = 4 * self.reg_max + self.nc
            if c != expected_features:
                raise ValueError(f"Prediction shape mismatch. Got {c} features, expected {expected_features}")
            
            # Reshape predictions to [B, H, W, 4*reg_max + nc]
            pred = pred.view(b, h, w, -1)
            
            # Split predictions
            pred_box = pred[..., :4*self.reg_max]  # [B, H, W, 4*reg_max]
            pred_cls = pred[..., 4*self.reg_max:]  # [B, H, W, nc]

            # Initialize targets
            tgt_mask = torch.zeros((b, h, w), device=device, dtype=torch.bool)
            tgt_box = torch.zeros((b, h, w, 4), device=device)
            tgt_cls = torch.zeros((b, h, w, self.nc), device=device)
            
            # Assign targets
            for t in targets:
                img_idx = int(t[0].item())
                if len(t) < 6:  # Skip invalid targets
                    continue
                    
                class_id = int(t[1].item())
                x, y, w_box, h_box = t[2:6]  # Normalized [0,1]
                
                # Calculate grid cell
                grid_x = min(int(x * w), w-1)
                grid_y = min(int(y * h), h-1)
                
                # Assign to grid cell
                tgt_mask[img_idx, grid_y, grid_x] = True
                tgt_box[img_idx, grid_y, grid_x] = torch.tensor([x, y, w_box, h_box], device=device)
                tgt_cls[img_idx, grid_y, grid_x, class_id] = 1.0

            # Compute losses only where targets exist
            if tgt_mask.any():
                # Get active predictions and targets
                pred_box_active = pred_box[tgt_mask]  # [M, 4*reg_max]
                tgt_box_active = tgt_box[tgt_mask]   # [M, 4]
                
                # Verify divisible by 4*reg_max
                if pred_box_active.numel() % (4 * self.reg_max) != 0:
                    print(f"Warning: Prediction elements {pred_box_active.numel()} not divisible by {4 * self.reg_max}")
                    continue
                
                # Convert targets to distribution targets
                tgt_ltrb = self._xywh_to_ltrb(tgt_box_active)  # [M, 4]
                tgt_ltrb = tgt_ltrb * (self.reg_max - 1)  # Scale to reg_max range
                tgt_ltrb = tgt_ltrb.clamp(0, self.reg_max - 1 - 1e-6)  # CLAMP TO VALID RANGE
                
                # DFL loss - SAFE RESHAPING
                if self.use_dfl and pred_box_active.numel() > 0:
                    try:
                        # First ensure correct number of elements
                        M = pred_box_active.shape[0]
                        pred_box_active = pred_box_active.view(M, 4, self.reg_max)
                        
                        # Then flatten for DFL loss
                        pred_box_active = pred_box_active.view(-1, self.reg_max)
                        tgt_ltrb = tgt_ltrb.view(-1)
                        
                        dfl_loss += self._df_loss(pred_box_active, tgt_ltrb)
                    except Exception as e:
                        print(f"DFL reshape error: {e}")
                        print(f"Input shape: {pred_box_active.shape}")
                        continue
                
                # Convert predictions to box coordinates
                pred_dist = F.softmax(pred_box_active.view(-1, 4, self.reg_max), dim=-1)
                pred_xywh = (pred_dist * torch.arange(self.reg_max, device=device)).sum(-1)
                
                # CIoU loss
                iou = bbox_iou(pred_xywh, tgt_box_active, CIoU=True)
                box_loss += (1.0 - iou).mean()

                # Classification loss
                cls_loss += F.binary_cross_entropy_with_logits(
                    pred_cls[tgt_mask], tgt_cls[tgt_mask], reduction='mean'
                )

        # Apply weights
        box_loss *= self.box_weight
        cls_loss *= self.cls_weight
        dfl_loss *= self.dfl_weight
        total_loss = box_loss + cls_loss + dfl_loss

        return box_loss, cls_loss, dfl_loss, total_loss

    def _xywh_to_ltrb(self, xywh):
        """Convert xywh to ltrb format (left, top, right, bottom)."""
        x, y, w, h = xywh.unbind(-1)
        return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=-1)

    def _df_loss(self, pred_dist, target):
        # Add numerical stability
        pred_dist = F.softmax(pred_dist, dim=-1)  # Ensure proper distribution
        pred_dist = torch.clamp(pred_dist, min=1e-6, max=1-1e-6)  # Avoid log(0)
        
        target_left = target.long()
        target_right = target_left + 1
        weight_right = target - target_left.float()
        weight_left = 1 - weight_right
        
        # Additional clamping
        target_left = target_left.clamp(0, self.reg_max-1)
        target_right = target_right.clamp(0, self.reg_max-1)
        
        loss_left = -torch.log(pred_dist.gather(1, target_left.unsqueeze(1))).squeeze(1) * weight_left
        loss_right = -torch.log(pred_dist.gather(1, target_right.unsqueeze(1))).squeeze(1) * weight_right
        
        return (loss_left + loss_right).mean()
    

def custom_collate_fn(batch):
    images = []
    targets_list = []

    for i, (img, img_targets) in enumerate(batch):
        images.append(img)

        # img_targets should be a tensor [N_img, 5] or list of lists
        # where each row is [class_label, cx, cy, w, h] for that image.

        if img_targets is not None and len(img_targets) > 0:
            # Ensure img_targets is a tensor
            if not isinstance(img_targets, torch.Tensor):
                 img_targets = torch.tensor(img_targets, dtype=torch.float32)

            # Add the image_id (batch index) as the first column
            image_id_col = torch.full((img_targets.shape[0], 1), fill_value=i, dtype=torch.float32)
            targets_with_id = torch.cat([image_id_col, img_targets], dim=1) # Shape [N_img, 6]
            targets_list.append(targets_with_id)
        else:
             # If no targets in this image, append an empty tensor with the correct number of columns
             targets_list.append(torch.empty(0, 6, dtype=torch.float32)) # Shape [0, 6]

    # Stack images into a batch tensor
    images = torch.stack(images, 0)

    # Concatenate all individual target tensors from the list
    # This will result in a single tensor [Total_Ground_Truth_Boxes_in_Batch, 6]
    # If targets_list contains only empty tensors, this will result in torch.empty(0, 6)
    if len(targets_list) > 0:
        targets = torch.cat(targets_list, 0)
    else:
        targets = torch.empty(0, 6, dtype=torch.float32) # Handle case where batch is empty or all images have no targets

    return images, targets.to(images.device) # Move targets to the same device as images


    
def check_model_compatibility(model, pretrained_path):
    pretrained = torch.load(pretrained_path, map_location='cpu')
    if isinstance(pretrained, nn.Module):
        pretrained = pretrained.state_dict()
    elif isinstance(pretrained, dict) and 'model' in pretrained:
        pretrained = pretrained['model']
    
    model_keys = set(model.state_dict().keys())
    pretrained_keys = set(pretrained.keys())
    
    print(f"Common layers: {len(model_keys & pretrained_keys)}")
    print(f"Missing in pretrained: {len(model_keys - pretrained_keys)}")
    print(f"Extra in pretrained: {len(pretrained_keys - model_keys)}")

def load_pretrained_weights(new_model, pretrained_path, verbose=True):
    """Improved weight loading function that handles model objects"""
    device = next(new_model.parameters()).device
    
    # Load pretrained weights
    pretrained = torch.load(pretrained_path, map_location='cpu')
    print(type(pretrained))  # Should show either dict or DetectionModel
    if isinstance(pretrained, dict):
        print(pretrained.keys())  # Show available keys
    
    # Handle different formats
    if isinstance(pretrained, nn.Module):
        # Direct model object
        pretrained_state = pretrained.state_dict()
    elif isinstance(pretrained, dict):
        if 'model' in pretrained:
            # Check if 'model' is a state dict or model object
            if isinstance(pretrained['model'], dict):
                pretrained_state = pretrained['model']
            elif hasattr(pretrained['model'], 'state_dict'):
                pretrained_state = pretrained['model'].state_dict()
            else:
                raise ValueError("Unsupported 'model' format in checkpoint")
        elif 'state_dict' in pretrained:
            pretrained_state = pretrained['state_dict']
        else:
            pretrained_state = pretrained
    else:
        raise ValueError("Unsupported pretrained format")
    
    # Convert state dict keys
    new_state = new_model.state_dict()
    matched, missing = 0, 0
    
    for name, param in new_state.items():
        if name in pretrained_state:
            if param.shape == pretrained_state[name].shape:
                param.data.copy_(pretrained_state[name])
                matched += 1
            elif len(param.shape) == len(pretrained_state[name].shape):
                # Handle partial loading for conv layers
                if len(param.shape) >= 2:
                    min_dim0 = min(param.shape[0], pretrained_state[name].shape[0])
                    min_dim1 = min(param.shape[1], pretrained_state[name].shape[1])
                    param.data[:min_dim0, :min_dim1] = pretrained_state[name][:min_dim0, :min_dim1]
                    if verbose:
                        print(f"Partially loaded {name} ({pretrained_state[name].shape} -> {param.shape})")
                    matched += 1
                else:
                    if verbose:
                        print(f"Shape mismatch for {name}: {pretrained_state[name].shape} vs {param.shape}")
                    missing += 1
            else:
                if verbose:
                    print(f"Shape mismatch for {name}: {pretrained_state[name].shape} vs {param.shape}")
                missing += 1
        else:
            if verbose:
                print(f"Missing key: {name}")
            missing += 1
    
    if verbose:
        print(f"Loaded {matched}/{len(new_state)} layers, {missing} missing")
    
    new_model.to(device)
    return new_model


def transfer_learning(args):
    # Initialize model
    model = DetectionModel(variant=args.variant, num_classes=num_classes)
    
    # Load pretrained weights
    if args.pretrained:
        # check_model_compatibility(model, args.pretrained)
        model = load_pretrained_weights(model, args.pretrained)
    
    # # Verify loaded weights
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")
    #     # param.requires_grad=True
    
    # Freeze backbone if specified
    print(f"Freezing backbone: {args.freeze_backbone}")
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if any(f'model.{i}.' in name for i in range(11)):  # First 11 layers are backbone
                param.requires_grad = False
        print(f"Froze {sum(1 for p in model.parameters() if not p.requires_grad)} backbone parameters")
    
    # Create dataset and dataloader
    train_dataset = YOLODataset(images_dir=os.path.join(args.data_dir, 'images', 'train'),
        labels_dir=os.path.join(args.data_dir, 'labels', 'train'),
        img_size=args.img_size, transform=transform
    )
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    
    # img, targets = train_dataset[0]
    # print("Normalized targets:", targets)
        
    # Optimizer and scheduler
    # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)
    # criterion = YOLOLoss(device=args.device, num_classes=num_classes, strides=[8, 16, 32], reg_max=16)
    criterion = YOLOLoss2(nc=num_classes, reg_max=16)
    
    # Training loop
    model.train()
    model.to(args.device)
    
    for epoch in range(args.epochs):
        for i, (images, targets) in enumerate(dataloader):
            start_time = time.time()
            images = images.to(args.device)
            targets = [t.to(args.device) for t in targets]
            
            # Forward pass
            outputs = model(images)

            # f = open(f"training_{args.variant}_{epoch}_{i}.txt", "w")
            # f.write("Targets Data:\n")
            # for i, target in enumerate(targets):                
            #     f.write(str(i))
            #     f.write(", ")
            #     f.write(str(target))    
            #     f.write("\n")
            # f.write("\n\nOutputs Data:\n")
            # for i,output in enumerate(outputs):
            #     f.write(f"\n\nOutput {i}:\n")
            #     f.write(str(output.shape) + "\n")
            #     for j,o in enumerate(output):
            #         f.write(f"Output {i}_{j}:\n")
            #         f.write(str(o.shape) + "\n")
            #         f.write(str(o) + "\n")
            # f.close()
            
            # Compute loss
            box_loss, cls_loss, dfl_loss, total_loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            # total_loss.requires_grad = True

            # box_loss.requires_grad = True
            box_loss.backward(retain_graph=True)
            cls_loss.backward(retain_graph=True)
            dfl_loss.backward()
            # total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            end_time = time.time()
            
            # if i % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Batch {i+1}/{dataloader.__len__()}, "
                    f"Box: {box_loss.item():.4f}, Cls: {cls_loss.item():.4f}, "
                    f"DFL: {dfl_loss.item():.4f}, Total: {total_loss.item():.4f}, Time: {end_time - start_time:.4f}s")
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch+1) % args.save_interval == 0:
            torch.save(model.state_dict(), f"checkpoint_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='m', help='Model variant')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone layers')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['EPOCHS'], help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['BATCH_SIZE'], help='Batch size')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['LEARNING_RATE'], help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=TRAIN_CONFIG['WEIGHT_DECAY'], help='Weight decay')
    parser.add_argument('--save_interval', type=int, default=TRAIN_CONFIG['SAVE_INTERVAL'], help='Save interval in epochs')
    parser.add_argument('--output', type=str, default='custom_yolo_final.pth', help='Output model path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Root directory of dataset')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--workers', type=int, default=MODEL_CONFIG['NUM_WORKERS'], help='Number of data loader workers')
    
    args = parser.parse_args()
    transfer_learning(args)