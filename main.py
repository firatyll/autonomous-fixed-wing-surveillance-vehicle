"""
Main Application
Mini Talon VTAIL Camera Viewer - RGB + Thermal Side by Side
"""

import cv2
import numpy as np
import time
from camera_stream import CameraStream

RGB_CAM_TOPIC = "/world/runway/model/mini_talon_vtail/link/base_link/sensor/camera/image"
THERMAL_CAM_TOPIC = "/thermal_camera"

WINDOW_NAME = "Mini Talon Cameras"
DISPLAY_SCALE = 0.5


def main():
    rgb_camera = CameraStream(topic=RGB_CAM_TOPIC)
    thermal_camera = CameraStream(topic=THERMAL_CAM_TOPIC)
    
    if not rgb_camera.start():
        print("Failed to subscribe to RGB camera topic!")
        return
    
    if not thermal_camera.start():
        print("Failed to subscribe to thermal camera topic!")
        return
    
    try:
        while True:
            rgb_frame = rgb_camera.frame
            thermal_frame = thermal_camera.frame
            
            display_frame = None
            
            if rgb_frame is not None and thermal_frame is not None:
                rgb_h = rgb_frame.shape[0]
                thermal_h, thermal_w = thermal_frame.shape[:2]
                scale = rgb_h / thermal_h
                thermal_resized = cv2.resize(thermal_frame, (int(thermal_w * scale), rgb_h))
                
                display_frame = np.hstack((rgb_frame, thermal_resized))
            elif rgb_frame is not None:
                display_frame = rgb_frame
            elif thermal_frame is not None:
                display_frame = thermal_frame
            
            if display_frame is not None:
                h, w = display_frame.shape[:2]
                display_frame = cv2.resize(display_frame, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
                cv2.imshow(WINDOW_NAME, display_frame)
            
            cv2.waitKey(1)
            time.sleep(0.001)
    except KeyboardInterrupt:
        rgb_camera.stop()
        thermal_camera.stop()


if __name__ == "__main__":
    main()