import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FusedDetection:
    """Represents a confirmed fire detection from both sensors."""
    rgb_center: Tuple[int, int]
    thermal_center: Tuple[int, int]
    rgb_box: Tuple[int, int, int, int]
    thermal_box: Tuple[int, int, int, int]
    confidence: float
    thermal_temp: float


class SensorFusion:
    def __init__(self, position_threshold: float = 0.15):

        self._position_threshold = position_threshold
        self._confirmed_fire = False
    
    @property
    def fire_confirmed(self) -> bool:
        return self._confirmed_fire
    
    def fuse(self, 
             rgb_detections: List, 
             thermal_detections: List,
             rgb_frame_size: Tuple[int, int],
             thermal_frame_size: Tuple[int, int]) -> Tuple[List[FusedDetection], List[int]]:

        fused = []
        confirmed_rgb_indices = []
        
        if not rgb_detections or not thermal_detections:
            self._confirmed_fire = False
            return fused, confirmed_rgb_indices
        
        rgb_w, rgb_h = rgb_frame_size
        thermal_w, thermal_h = thermal_frame_size
        
        potential_matches = []
        
        for i, rgb_det in enumerate(rgb_detections):
            rgb_norm_x = rgb_det.center[0] / rgb_w
            rgb_norm_y = rgb_det.center[1] / rgb_h
            
            for j, thermal_det in enumerate(thermal_detections):
                thermal_norm_x = thermal_det.center[0] / thermal_w
                thermal_norm_y = thermal_det.center[1] / thermal_h
                
                dx = abs(rgb_norm_x - thermal_norm_x)
                dy = abs(rgb_norm_y - thermal_norm_y)
                
                if dx < self._position_threshold and dy < self._position_threshold:
                    confidence = 1.0 - (dx + dy) / (2 * self._position_threshold)
                    
                    match_obj = FusedDetection(
                        rgb_center=rgb_det.center,
                        thermal_center=thermal_det.center,
                        rgb_box=(rgb_det.x, rgb_det.y, rgb_det.width, rgb_det.height),
                        thermal_box=(thermal_det.x, thermal_det.y, thermal_det.width, thermal_det.height),
                        confidence=confidence,
                        thermal_temp=thermal_det.max_temp
                    )
                    potential_matches.append({
                        'confidence': confidence,
                        'rgb_idx': i,
                        'thermal_idx': j,
                        'match': match_obj
                    })
        
        potential_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        used_rgb = set()
        used_thermal = set()
        
        for m in potential_matches:
            if m['rgb_idx'] not in used_rgb and m['thermal_idx'] not in used_thermal:
                used_rgb.add(m['rgb_idx'])
                used_thermal.add(m['thermal_idx'])
                
                fused.append(m['match'])
                confirmed_rgb_indices.append(m['rgb_idx'])
        
        self._confirmed_fire = len(fused) > 0
        return fused, confirmed_rgb_indices
    
    def draw_warning(self, frame: np.ndarray, fused_detections: List[FusedDetection]) -> np.ndarray:
        """Draw FIRE DETECTED warning on frame."""
        if not fused_detections:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 150), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
        
        best = max(fused_detections, key=lambda d: d.confidence)
        text = f"FIRE DETECTED - {best.thermal_temp:.0f}C - Confidence: {best.confidence:.0%}"
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        
        cv2.putText(result, text, (text_x, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return result
