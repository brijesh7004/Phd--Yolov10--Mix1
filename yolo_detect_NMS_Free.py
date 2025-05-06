import torch
import cv2
import numpy as np
from yolov10_model import DetectionModel
from utils import non_max_suppression, scale_coords  # You'll need to implement these
import argparse
import time

from credentials import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, DATA_DIR
num_classes = MODEL_CONFIG['NUM_CLASSES']

class NMSFreeDetector:
    def __init__(self, model_path, variant='m', conf_thresh=0.25, iou_thresh=0.45, device='cuda'):
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Load model
        self.model = DetectionModel(variant=variant, num_classes=num_classes).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # Get stride from model (you may need to adjust this based on your model)
        self.stride = 32  # Example stride, adjust based on your model
        
    def preprocess(self, image, img_size=640):
        """Preprocess image for inference"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Calculate ratio
        r = min(img_size / h, img_size / w)
        new_h, new_w = int(h * r), int(w * r)
        pad_h = img_size - new_h
        pad_w = img_size - new_w
        
        # Resize and pad
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=114)
        
        # Convert to tensor
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # 0-255 to 0.0-1.0
        img = img.unsqueeze(0)  # Add batch dimension
        
        # Return both the ratio and pad values
        ratios = (r, (pad_w, pad_h))  # Now returns a tuple of (ratio, pad)
        return img, (h, w), ratios
    
    def postprocess(self, preds, img_shape, ratios):
        """Postprocess predictions (NMS-free)"""
        # Assuming preds is a list of tensors from different detection layers
        # You'll need to adjust this based on your model's output format
        
        # For YOLOv10 NMS-free output, we can directly use the one-to-one branches
        detections = []
        for i, pred in enumerate(preds):
            # pred shape: [batch, num_anchors, num_classes + 4]
            # Process each prediction layer
            if isinstance(pred, list):
                pred = pred[0]  # Take the first output (one-to-one branch)
            
            # Filter by confidence threshold
            pred = pred[pred[:, 4] > self.conf_thresh]
            
            if len(pred) == 0:
                continue
                
            # Get boxes and scores
            boxes = pred[:, :4]
            scores = pred[:, 4:5] * pred[:, 5:]  # conf * cls_prob
            
            # Scale boxes to original image size
            boxes = scale_coords(boxes, img_shape, ratios)
            
            # Get class IDs and scores
            class_ids = torch.argmax(scores, dim=1)
            class_scores = torch.max(scores, dim=1).values
            
            # Combine results
            for box, class_id, score in zip(boxes, class_ids, class_scores):
                detections.append({
                    'box': box.cpu().numpy(),
                    'class_id': class_id.item(),
                    'score': score.item()
                })
        
        return detections
    
    def detect(self, image):
        # Preprocessing remains the same
        img_tensor, img_shape, ratios = self.preprocess(image)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
            # Handle outputs based on your model's architecture
            if isinstance(outputs, (list, tuple)):
                # Process multiple outputs if your model has them
                detections = []
                for output in outputs:
                    detections.extend(self.process_output(output, img_shape, ratios, img_tensor))
            else:
                detections = self.process_output(outputs, img_shape, ratios, img_tensor)
                
        return detections
    
    def process_output(self, output, img_shape, ratios, img_tensor):
        """Process model output with 576 channels"""
        output = output[0]  # Remove batch dimension [576, height, width]
        channels, height, width = output.shape
        
        # Calculate how many classes this output represents
        # Assuming format: [box_coords (4) + objectness (1) + classes (N)]
        num_classes = channels - 5  # Total channels minus 5 (4 box + 1 obj)
        
        # Reshape output
        num_anchors = height * width
        output = output.permute(1, 2, 0).contiguous().view(num_anchors, channels)
        
        # Split into components
        boxes = output[:, :4]          # [num_anchors, 4] (x1,y1,x2,y2)
        obj_scores = output[:, 4]      # [num_anchors] (objectness)
        cls_scores = output[:, 5:]     # [num_anchors, num_classes]
        
        # Apply activations
        obj_scores = obj_scores.sigmoid()
        cls_scores = cls_scores.sigmoid()
        
        # Combine scores
        scores = obj_scores.unsqueeze(1) * cls_scores  # [num_anchors, num_classes]
        
        # Filter by confidence and valid boxes
        max_scores, class_ids = scores.max(dim=1)
        valid_mask = (max_scores > self.conf_thresh) & \
                    (boxes[:, 2] > boxes[:, 0]) & \
                    (boxes[:, 3] > boxes[:, 1])
        
        filtered_boxes = boxes[valid_mask]
        filtered_scores = max_scores[valid_mask]
        filtered_class_ids = class_ids[valid_mask]
        
        if len(filtered_boxes) == 0:
            return []
        
        # Scale and clip boxes
        filtered_boxes = scale_coords(img_tensor.shape[2:], filtered_boxes, img_shape, ratios)
        filtered_boxes[:, [0, 2]] = filtered_boxes[:, [0, 2]].clamp(0, img_shape[1])
        filtered_boxes[:, [1, 3]] = filtered_boxes[:, [1, 3]].clamp(0, img_shape[0])
        
        # Convert to detection format
        return [{
            'box': box.cpu().numpy().tolist(),
            'score': score.item(),
            'class_id': class_id.item()
        } for box, score, class_id in zip(filtered_boxes, filtered_scores, filtered_class_ids)]
    
def visualize(image, detections, class_names):
    """Visualize detections on image"""
    for det in detections:
        box = det['box']
        class_id = det['class_id']
        score = det['score']

        if class_id>num_classes:
            class_id = class_id // num_classes
                
        box[0], box[2] = box[0] * image.shape[1], box[2] * image.shape[1]
        box[1], box[3] = box[1] * image.shape[0], box[3] * image.shape[0]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print(x1, x2, y1, y2, class_id, score)
        
        # label = f"{class_names[class_id]}: {score:.2f}"
        # cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--variant', type=str, default='m', help='Model variant')
    parser.add_argument('--source', type=str, required=True, help='Image or video source')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--output', type=str, default='output.jpg', help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = NMSFreeDetector(args.model, variant=args.variant, conf_thresh=args.conf, device=args.device)
    
    # Load image
    image = cv2.imread(args.source)
    if image is None:
        raise ValueError(f"Could not load image from {args.source}")
    
    # Detect objects
    detections = detector.detect(image)
    print(detections)
    
    # Visualize results
    class_names = MODEL_CONFIG['CLASS_NAMES2']
    result = visualize(image.copy(), detections, class_names)
    
    # Save or show results
    cv2.imwrite(args.output, result)
    print(f"Results saved to {args.output}")
    print(f"Detected {len(detections)} objects")