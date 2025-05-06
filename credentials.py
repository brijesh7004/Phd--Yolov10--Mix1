"""
YOLOv10 Configuration
"""

import torch
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
# WEIGHTS_DIR = BASE_DIR / 'weights'
# RESULTS_DIR = BASE_DIR / 'results'

# # Ensure directories exist
# WEIGHTS_DIR.mkdir(exist_ok=True)
# RESULTS_DIR.mkdir(exist_ok=True)

# # Default model weights
# MODEL_WEIGHTS = str(WEIGHTS_DIR / "yolov10b.pt")
MODEL_WEIGHTS_TEST = str("best_yolov10.pth")
# MODEL_WEIGHTS_TRAINING = str("best_yolov10_train.pth")
MODEL_WEIGHTS_TRAINING = str("best_yolov10_pc.pth")

# Device configuration
DEVICE_CONFIG = {
    "CUDA": True,      # Enable CUDA if available
    "AMP": True,       # Enable Automatic Mixed Precision
}

# Model configuration
MODEL_CONFIG = {
    # COCO class names
    # "CLASS_NAMES": [ "car", "motorcycle", "person" ],
    "CLASS_NAMES_80": [
       "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
       "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
       "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
       "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
       "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
       "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
       "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
       "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
       "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
       "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
   ],
    "CLASS_NAMES": [ "Person", "Motorcycle", "Car", "Truck", "Bus"],
    "CLASS_NAMES2": ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck'],
    "NUM_CLASSES": 5,
    
    # Anchor configuration (3 anchors per scale, in [w, h] format)
    "ANCHORS": [
        [[10, 13], [16, 30], [33, 23]],       # Small objects
        [[30, 61], [62, 45], [59, 119]],      # Medium objects
        [[116, 90], [156, 198], [373, 326]]   # Large objects
    ],
    
    # Detection parameters
    "STRIDES": [8, 16, 32],
    "ANCHOR_PER_SCALE": 3,
    "IMG_SIZE": 640,
    "CONF_THRESHOLD": 0.25,
    "IOU_THRESHOLD": 0.45,
    "NUM_WORKERS": 4,
}

# Training configuration
TRAIN_CONFIG = {
    'MODEL_VARIANT': 'm',
    'BATCH_SIZE': 16,
    'EPOCHS': 100,
    'LEARNING_RATE': 0.001,
    'WEIGHT_DECAY': 0.0005,
    'MOMENTUM': 0.937,
    'WARMUP_EPOCHS': 3,
    'LOG_INTERVAL': 1,
    'SAVE_INTERVAL': 1,
    'EVAL_INTERVAL': 1,
    'PRETRAINED_WEIGHTS': None,  # Path to pretrained weights if any
    # 'SAVE_DIR': str(WEIGHTS_DIR),
    'RESUME': False,  # Whether to resume from last checkpoint
}

# Data configuration
DATA_CONFIG = {
    'IMG_DIR': str(DATA_DIR / 'images'),
    'LABEL_DIR': str(DATA_DIR / 'labels'),

    'TRAIN_IMG_DIR': str(DATA_DIR / 'images' / 'train'),
    'VAL_IMG_DIR': str(DATA_DIR / 'images' / 'val'),
    'TEST_IMG_DIR': str(DATA_DIR / 'images' / 'test'),
    'TRAIN_LABEL_DIR': str(DATA_DIR / 'labels' / 'train'),
    'VAL_LABEL_DIR': str(DATA_DIR / 'labels' / 'val'),
    'TEST_LABEL_DIR': str(DATA_DIR / 'labels' / 'test'),
}
