"""
detector.py - Computer Vision logic using YOLOv8

Detects 'cell phone' (Class 67) and 'person' (Class 0) in video frames.
Returns bounding box coordinates and class labels.
"""

from ultralytics import YOLO

# Classes of interest
PERSON_CLASS = 0
CELL_PHONE_CLASS = 67


class PhoneDetector:
    """Handles YOLOv8 object detection for phone and person."""

    def __init__(self, model_name='yolov8s.pt', confidence=0.25):
        """
        Initialize the YOLOv8 detector.

        Args:
            model_name: YOLOv8 model name (auto-downloads if not found).
            confidence: Minimum confidence threshold for detections.
        """
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame):
        """
        Run detection on a single frame.

        Args:
            frame: A BGR image (numpy array) from OpenCV.

        Returns:
            A dict with keys 'persons' and 'phones', each containing
            a list of bounding boxes in [x1, y1, x2, y2] format.
        """
        results = self.model(frame, conf=self.confidence, verbose=False)

        persons = []
        phones = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                if cls_id == PERSON_CLASS:
                    persons.append(coords)
                elif cls_id == CELL_PHONE_CLASS:
                    phones.append(coords)

        return {'persons': persons, 'phones': phones}
