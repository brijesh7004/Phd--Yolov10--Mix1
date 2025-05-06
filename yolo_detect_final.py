import collections
import torch
import cv2
import numpy as np
from time import time
from torchvision.transforms import functional as F


from yolov10_model import DetectionModel  # Assuming your model is in model.py
from credentials import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, DATA_DIR
num_classes = MODEL_CONFIG['NUM_CLASSES']

import torch
import cv2
import numpy as np
from time import time
from torchvision.transforms import functional as F

class FastYOLOv10Detector:
    def __init__(self, model_path, device='cpu', conf_thresh=0.25):
        self.device = torch.device(device)
        
        # Load model (handling both state_dict and full model)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            self.model = DetectionModel(variant=TRAIN_CONFIG['MODEL_VARIANT'], num_classes=num_classes)  # Adjust as needed
            self.model.load_state_dict(checkpoint)
        else:
            self.model = checkpoint
            
        self.model = self.model.float().eval()
        self.conf_thresh = conf_thresh
        self.img_size = 640
        self.nc = num_classes
        
    def preprocess(self, img):
        """Resize and normalize image while maintaining aspect ratio"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        r = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * r), int(w * r)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Pad to square
        top = (self.img_size - new_h) // 2
        bottom = self.img_size - new_h - top
        left = (self.img_size - new_w) // 2
        right = self.img_size - new_w - left
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                       cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert to tensor
        img_tensor = F.to_tensor(img_padded).unsqueeze(0).to(self.device)
        return img_tensor, (r, (left, top))
    
    def postprocess(self, preds, ratio, padding):
        """Convert model outputs to detections with proper tensor handling"""
        detections = []
        for i, pred in enumerate(preds):
            stride = 8 * (2 ** i)  # P3:8, P4:16, P5:32
            pred = pred.sigmoid()
            b, _, h, w = pred.shape
            
            # Create grid properly
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid = torch.stack((grid_x, grid_y), -1).to(pred.device)  # [H, W, 2]
            
            # Reshape predictions
            pred = pred.permute(0, 2, 3, 1)  # [B, H, W, C]
            box_dim = 4 + 1 + self.nc  # xywh + conf + classes
            
            # Split predictions
            pred_box = pred[..., :4]  # [B, H, W, 4]
            pred_conf = pred[..., 4:5]  # [B, H, W, 1]
            pred_cls = pred[..., 5:]  # [B, H, W, nc]
            
            # Decode boxes with proper broadcasting
            xy = (pred_box[..., :2] * 2 - 0.5 + grid) * stride  # [B, H, W, 2]
            wh = (pred_box[..., 2:4] * 2) ** 2 * stride  # [B, H, W, 2]
            
            # Combine coordinates
            boxes = torch.cat([
                xy.view(-1, 2),  # Flatten to [N, 2]
                wh.view(-1, 2)   # Flatten to [N, 2]
            ], dim=1)  # [N, 4]
            
            # Confidence filtering
            scores = (pred_conf * pred_cls.max(-1, keepdim=True)[0]).view(-1)  # [N]
            mask = scores > self.conf_thresh
            boxes = boxes[mask]
            scores = scores[mask]
            cls_ids = pred_cls.argmax(-1).view(-1)[mask]
            
            if len(boxes) == 0:
                continue
                
            # Rescale to original image
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padding[0]) / ratio
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padding[1]) / ratio
            
            detections.append(torch.cat([
                boxes,
                scores.unsqueeze(1),
                cls_ids.unsqueeze(1)
            ], dim=1))
        
        return torch.cat(detections).cpu().numpy() if len(detections) > 0 else np.zeros((0, 6))

    def detect(self, img):
        """Run end-to-end detection"""
        img_tensor, (ratio, padding) = self.preprocess(img)
        
        with torch.no_grad():
            preds = self.model(img_tensor)
        
        return self.postprocess(preds, ratio, padding)
    
if __name__ == "__main__":
    # Initialize detector
    detector = FastYOLOv10Detector("checkpoint_10.pth", device='cpu')
    
    # Load image
    filePath = "data/images/test/frame_0990_jpg.rf.a3223422e734a57442ee34a58d24d4b4.jpg"
    img = cv2.imread(filename=filePath)
    if img is None:
        raise FileNotFoundError("Image not found")
    
    # Run detection
    detections = detector.detect(img)
    
    # Visualize results
    for det in detections:
        x, y, w, h, conf, cls_id = det
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{int(cls_id)}:{conf:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    cv2.imwrite("output1.jpg", img)
    # cv2.imshow("Detection", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()