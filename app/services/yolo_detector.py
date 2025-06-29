from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
from typing import Tuple, List
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class YOLODetector:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YOLODetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        try:
            self.model = self._load_model()
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model():
        try:
            return YOLO('yolov8n.pt')  # Load YOLOv8 nano model
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise

    def detect(self, image_bytes: bytes, conf_threshold: float = 0.5) -> Tuple[List[dict], np.ndarray]:
        try:
            # Validate image
            image = self._validate_and_prepare_image(image_bytes)
            
            # Run inference
            results = self.model(image, conf=conf_threshold)
            result = results[0]  # Get first result

            # Process detections
            detections = []
            for box in result.boxes:
                cls = result.names[box.cls[0].item()]
                conf = box.conf[0].item()
                xyxy = box.xyxy[0].tolist()
                
                detections.append({
                    'class': cls,
                    'confidence': float(conf),
                    'box': xyxy
                })

            # Get annotated image
            annotated_image = result.plot()

            return detections, annotated_image

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            raise

    def _validate_and_prepare_image(self, image_bytes: bytes) -> Image.Image:
        try:
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Check image size
            max_size = 1920  # Maximum dimension
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.LANCZOS)
                
            return image
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

def detect_objects(image_bytes: bytes, keyword: str, conf_threshold: float = 0.5) -> Tuple[bool, List[str], np.ndarray]:
    try:
        # Get singleton instance
        detector = YOLODetector()
        detections, annotated_image = detector.detect(image_bytes, conf_threshold)

        # Extract classes with confidence scores
        detected_classes = [
            f"{det['class']} ({det['confidence']:.2f})"
            for det in detections
        ]

        # Clean and normalize the keyword and detected classes for comparison
        keyword = keyword.lower().strip()
        detected_classes_lower = [det['class'].lower().strip() for det in detections]

        # Special cases mapping with more categories
        keyword_mapping = {
            'football': ['sports ball', 'ball'],
            'soccer ball': ['sports ball', 'ball'],
            'ball': ['sports ball', 'ball', 'baseball', 'tennis ball'],
            'person': ['person', 'human', 'pedestrian'],
            'car': ['car', 'vehicle', 'automobile'],
            # Add more mappings as needed
        }

        # Check if keyword matches any detected class
        matched = keyword in detected_classes_lower
        if not matched and keyword in keyword_mapping:
            matched = any(cls in detected_classes_lower for cls in keyword_mapping[keyword])

        return matched, detected_classes, annotated_image

    except Exception as e:
        logger.error(f"Error in detect_objects: {e}")
        raise