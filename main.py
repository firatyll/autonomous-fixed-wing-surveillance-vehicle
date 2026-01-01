"""
Main Application
Mini Talon VTAIL Camera Viewer - RGB + Thermal Side by Side with GPS Telemetry
"""

import cv2
import numpy as np
import time
from camera_stream import CameraStream
from telemetry import Telemetry
from thermal_detector import ThermalDetector

RGB_CAM_TOPIC = "/world/runway/model/mini_talon_vtail/link/base_link/sensor/camera/image"
THERMAL_CAM_TOPIC = "/thermal_camera"
MAVLINK_CONNECTION = "udp:127.0.0.1:14550"

WINDOW_NAME = "Mini Talon Cameras"
DISPLAY_SCALE = 0.5
INFO_BAR_HEIGHT = 40

THERMAL_TEMP_THRESHOLD = 500.0


def draw_telemetry_bar(frame: np.ndarray, gps: dict) -> np.ndarray:
    """Draw GPS telemetry info bar below the frame."""
    h, w = frame.shape[:2]
    
    info_bar = np.zeros((INFO_BAR_HEIGHT, w, 3), dtype=np.uint8)
    info_bar[:] = (40, 40, 40)
    
    text = (
        f"LAT: {gps['lat']:.6f}  "
        f"LON: {gps['lon']:.6f}  "
        f"ALT: {gps['alt']:.1f}m  "
        f"HDG: {gps['heading']:.0f}°  "
        f"GS: {gps['groundspeed']:.1f}m/s  "
        f"SAT: {gps['satellites']} (Fix:{gps['fix_type']})"
    )
    
    cv2.putText(info_bar, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return np.vstack((frame, info_bar))


def main():
    rgb_camera = CameraStream(topic=RGB_CAM_TOPIC)
    thermal_camera = CameraStream(topic=THERMAL_CAM_TOPIC)
    telemetry = Telemetry(connection_string=MAVLINK_CONNECTION)
    thermal_detector = ThermalDetector(temp_threshold=THERMAL_TEMP_THRESHOLD)
    
    if not rgb_camera.start():
        print("Failed to subscribe to RGB camera topic!")
        return
    
    if not thermal_camera.start():
        print("Failed to subscribe to thermal camera topic!")
        return
    
    if not telemetry.start():
        print("Warning: Telemetry not available, continuing without GPS...")
    
    try:
        while True:
            rgb_frame = rgb_camera.frame
            thermal_frame = thermal_camera.frame
            gps = telemetry.gps
            
            display_frame = None
            thermal_display = thermal_frame
            
            if thermal_frame is not None:
                detections = thermal_detector.detect(thermal_frame)
                if detections:
                    thermal_display = thermal_detector.draw_detections(thermal_frame, detections)
            
            if rgb_frame is not None and thermal_display is not None:
                display_frame = np.hstack((rgb_frame, thermal_display))
            elif rgb_frame is not None:
                display_frame = rgb_frame
            elif thermal_display is not None:
                display_frame = thermal_display
            
            if display_frame is not None:
                h, w = display_frame.shape[:2]
                display_frame = cv2.resize(display_frame, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
                display_frame = draw_telemetry_bar(display_frame, gps)
                cv2.imshow(WINDOW_NAME, display_frame)
            
            cv2.waitKey(1)
            time.sleep(0.001)
    except KeyboardInterrupt:
        rgb_camera.stop()
        thermal_camera.stop()
        telemetry.stop()


if __name__ == "__main__":
    main()