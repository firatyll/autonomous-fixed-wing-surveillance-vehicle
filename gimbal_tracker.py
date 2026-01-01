import time
import subprocess
from typing import Tuple, Optional
import cv2
import numpy as np


class GimbalTracker:
    """
    Controls gimbal to track confirmed fire targets.
    
    Features:
    - 3-second lock timer before tracking starts
    - P-controller with fine-tuning in deadband
    - Reset to center when target lost
    - Visual overlay (crosshair + status)
    """
    
    def __init__(self,
                 pan_topic: str = "/camera/pan",
                 tilt_topic: str = "/camera/tilt",
                 lock_duration: float = 3.0):
        # Topics
        self.pan_topic = pan_topic
        self.tilt_topic = tilt_topic
        
        # Limits (radians)
        self.pan_left = 1.57
        self.pan_right = -1.57
        self.tilt_up = -1.57
        self.tilt_down = 1.57
        
        # P-controller gains
        self.pan_kp = 0.05
        self.tilt_kp = 0.02
        
        # Dead zones
        self.pan_dead_zone = 0.03
        self.pan_unlock_zone = 0.05
        self.tilt_dead_zone = 0.08
        self.tilt_unlock_zone = 0.12
        
        # Fine-tune threshold
        self.fine_tune_threshold = 0.01
        self.fine_tune_gain = 0.3
        
        # Lock timer
        self.lock_duration = lock_duration
        self.target_first_seen: Optional[float] = None
        
        # State
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.is_locked = False
        self.is_yaw_centered = False
        self.is_pitch_centered = False
        self.frame_count = 0
    
    @property
    def is_centered(self) -> bool:
        """Returns True when target is centered (both axes in deadband)."""
        return self.is_locked and self.is_yaw_centered and self.is_pitch_centered
    
    @property
    def pan(self) -> float:
        """Current pan angle in radians."""
        return self.current_pan
    
    @property
    def tilt(self) -> float:
        """Current tilt angle in radians."""
        return self.current_tilt
    
    def update(self, fused_detections: list, frame_shape: Tuple[int, int]) -> None:
        """
        Update gimbal tracking based on fused fire detections.
        Call every frame. Internally rate-limits to every 5 frames.
        
        Args:
            fused_detections: List of FusedDetection objects (confirmed fires)
            frame_shape: (height, width) of RGB frame
        """
        self.frame_count += 1
        
        # Only process every 3 frames
        if self.frame_count % 3 != 0:
            return
        
        if fused_detections:
            self._handle_target(fused_detections, frame_shape)
        else:
            self._handle_no_target()
    
    def _handle_target(self, fused_detections: list, frame_shape: Tuple[int, int]) -> None:
        """Handle when target is present."""
        target = max(fused_detections, key=lambda d: d.confidence)
        target_x, target_y = target.rgb_center
        
        # Start lock timer if new target
        if self.target_first_seen is None:
            self.target_first_seen = time.time()
            print("[GIMBAL] Target detected, starting 3s lock timer...")
        
        elapsed = time.time() - self.target_first_seen
        
        # Only track after lock duration
        if elapsed >= self.lock_duration:
            if not self.is_locked:
                self.is_locked = True
                print("[GIMBAL] LOCKED ON TARGET!")
            
            self._track(target_x, target_y, frame_shape)
    
    def _handle_no_target(self) -> None:
        """Handle when no target is present - reset to center."""
        if self.target_first_seen is not None:
            print("[GIMBAL] Target lost, resetting to center...")
            self.target_first_seen = None
            self.is_locked = False
            self.is_yaw_centered = False
            self.is_pitch_centered = False
            
            self.current_pan = 0.0
            self.current_tilt = 0.0
            self._send_command(self.pan_topic, 0.0)
            self._send_command(self.tilt_topic, 0.0)
    
    def _track(self, target_x: int, target_y: int, frame_shape: Tuple[int, int]) -> None:
        """Track target with P-controller."""
        height, width = frame_shape
        center_x = width / 2.0
        center_y = height / 2.0
        
        # Calculate normalized errors (-1 to +1)
        error_x = (target_x - center_x) / center_x
        error_y = (target_y - center_y) / center_y
        
        # ===== YAW (PAN) =====
        if self.is_yaw_centered:
            if abs(error_x) > self.pan_unlock_zone:
                self.is_yaw_centered = False
        else:
            if abs(error_x) <= self.pan_dead_zone:
                self.is_yaw_centered = True
        
        if not self.is_yaw_centered:
            # Normal correction
            new_pan = self.current_pan - (error_x * self.pan_kp)
            new_pan = max(self.pan_right, min(self.pan_left, new_pan))
            self.current_pan = round(new_pan, 3)
            self._send_command(self.pan_topic, self.current_pan)
        elif abs(error_x) > self.fine_tune_threshold:
            # Fine-tune within deadband
            new_pan = self.current_pan - (error_x * self.pan_kp * self.fine_tune_gain)
            new_pan = max(self.pan_right, min(self.pan_left, new_pan))
            self.current_pan = round(new_pan, 3)
            self._send_command(self.pan_topic, self.current_pan)
        
        # ===== PITCH (TILT) =====
        if self.is_pitch_centered:
            if abs(error_y) > self.tilt_unlock_zone:
                self.is_pitch_centered = False
        else:
            if abs(error_y) <= self.tilt_dead_zone:
                self.is_pitch_centered = True
        
        if not self.is_pitch_centered:
            new_tilt = self.current_tilt + (error_y * self.tilt_kp)
            new_tilt = max(self.tilt_up, min(self.tilt_down, new_tilt))
            self.current_tilt = round(new_tilt, 3)
            self._send_command(self.tilt_topic, self.current_tilt)
    
    def _send_command(self, topic: str, value: float) -> None:
        """Send command to Gazebo gimbal topic."""
        cmd = [
            "gz", "topic",
            "-t", topic,
            "-m", "gz.msgs.Double",
            "-p", f"data: {value:.4f}"
        ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tracking status overlay on frame.
        
        Args:
            frame: RGB frame to draw on
            
        Returns:
            Frame with overlay drawn
        """
        if frame is None:
            return frame
        
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        if self.is_locked:
            # LOCKED - green crosshair
            cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 3)
            cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 3)
            cv2.rectangle(frame, (cx - 75, cy - 75), (cx + 75, cy - 45), (0, 0, 0), -1)
            cv2.putText(frame, "LOCKED", (cx - 70, cy - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        elif self.target_first_seen is not None:
            # LOCKING - yellow crosshair
            remaining = max(0, self.lock_duration - (time.time() - self.target_first_seen))
            cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 255, 255), 2)
            cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 255, 255), 2)
            cv2.rectangle(frame, (cx - 110, cy - 75), (cx + 110, cy - 45), (0, 0, 0), -1)
            cv2.putText(frame, f"LOCKING {remaining:.1f}s", (cx - 100, cy - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        return frame
