import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FireDetection:
    """Represents a detected fire region."""
    x: int
    y: int
    width: int
    height: int
    area: int
    center: Tuple[int, int]


class FireDetector:
    """Detects fire-colored regions using HSV color filtering."""
    
    def __init__(self, min_area: int = 20):
        self._min_area = min_area
        
        self._lower_hsv = np.array([3, 120, 120])
        self._upper_hsv = np.array([25, 255, 255])
    
    def detect(self, frame: np.ndarray) -> List[FireDetection]:

        if frame is None:
            return []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, self._lower_hsv, self._upper_hsv)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            
            detections.append(FireDetection(
                x=x, y=y, width=w, height=h,
                area=area, center=center
            ))
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[FireDetection], 
                         confirmed_indices: List[int] = None) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        result = frame.copy()
        
        if confirmed_indices is None:
            confirmed_indices = []
        
        for i, det in enumerate(detections):
            is_confirmed = i in confirmed_indices
            
            if is_confirmed:
                color = (0, 255, 0)
                label = "ACTUAL FIRE"
            else:
                color = (0, 165, 255)
                label = "FALSE POSITIVE"
            
            cv2.rectangle(result, (det.x, det.y), 
                         (det.x + det.width, det.y + det.height), 
                         color, 2)
            
            cv2.putText(result, label, (det.x, det.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
