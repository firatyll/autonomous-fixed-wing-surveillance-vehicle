"""
Thermal Detection Module
Detects high temperature regions in thermal camera feed.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ThermalDetection:
    """Represents a detected hot region."""
    x: int
    y: int
    width: int
    height: int
    max_temp: float
    center: Tuple[int, int]


class ThermalDetector:
    """Detects high temperature regions in thermal images."""
    
    def __init__(self, temp_threshold: float = 500.0, min_area: int = 10):
        self._temp_threshold = temp_threshold
        self._min_area = min_area
        
        self._temp_min = 0.0
        self._temp_max = 1000.0
    
    def _pixel_to_temp(self, pixel_value: int) -> float:
        """Convert 8-bit pixel value (0-255) to temperature."""
        return self._temp_min + (pixel_value / 255.0) * (self._temp_max - self._temp_min)
    
    def _temp_to_pixel(self, temp: float) -> int:
        """Convert temperature to 8-bit pixel value."""
        normalized = (temp - self._temp_min) / (self._temp_max - self._temp_min)
        return int(np.clip(normalized * 255, 0, 255))
    
    def detect(self, thermal_frame: np.ndarray) -> List[ThermalDetection]:
        """
        Detect hot regions above threshold temperature.
        
        Args:
            thermal_frame: Grayscale thermal image (0-255)
            
        Returns:
            List of ThermalDetection objects
        """
        if thermal_frame is None:
            return []
        
        if len(thermal_frame.shape) == 3:
            gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = thermal_frame
        
        threshold_pixel = self._temp_to_pixel(self._temp_threshold)
        _, binary = cv2.threshold(gray, threshold_pixel, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            roi = gray[y:y+h, x:x+w]
            max_pixel = np.max(roi)
            max_temp = self._pixel_to_temp(max_pixel)
            
            center = (x + w // 2, y + h // 2)
            
            detections.append(ThermalDetection(
                x=x, y=y, width=w, height=h,
                max_temp=max_temp, center=center
            ))
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[ThermalDetection]) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        result = frame.copy()
        
        for det in detections:
            cv2.rectangle(result, (det.x, det.y), 
                         (det.x + det.width, det.y + det.height), 
                         (0, 0, 255), 2)
            
            label = f"FIRE {det.max_temp:.0f}C"
            cv2.putText(result, label, (det.x, det.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return result
