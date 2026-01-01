import cv2
import time
import numpy as np
from camera_stream import CameraStream
from thermal_detector import ThermalDetector
from fire_detector import FireDetector
from sensor_fusion import SensorFusion
from gimbal_tracker import GimbalTracker
from telemetry import Telemetry
from target_locator import TargetLocator

RGB_TOPIC = "/world/runway/model/mini_talon_vtail/link/camera_tilt_link/sensor/camera/image"
THERMAL_TOPIC = "/thermal_camera"
WINDOW = "Fire Detection - Fusion"
TH_THRESH = 500.0
MAX_WIDTH = 1280
WAYPOINT_INTERVAL = 5.0

def main():
    rgb_cam = CameraStream(RGB_TOPIC)
    th_cam = CameraStream(THERMAL_TOPIC)
    th_det = ThermalDetector(TH_THRESH)
    rgb_det = FireDetector()
    fusion = SensorFusion(0.15)
    gimbal_tracker = GimbalTracker()
    telemetry = Telemetry()
    target_locator = TargetLocator()

    rgb_cam.start()
    th_cam.start()
    telemetry.start()

    last_waypoint_time = 0.0
    target_lat, target_lon = None, None

    while True:
        rgb_frame = rgb_cam.frame
        th_frame = th_cam.frame

        if rgb_frame is None or th_frame is None:
            cv2.waitKey(1)
            continue

        th_detections = th_det.detect(th_frame)
        rgb_detections = rgb_det.detect(rgb_frame)

        rgb_h, rgb_w = rgb_frame.shape[:2]
        th_h, th_w = th_frame.shape[:2]

        fused, confirmed_rgb_indices = fusion.fuse(
            rgb_detections, th_detections,
            (rgb_w, rgb_h), (th_w, th_h)
        )

        gimbal_tracker.update(fused, (rgb_h, rgb_w))

        if gimbal_tracker.is_centered:
            gps = telemetry.gps
            target_location = target_locator.locate_target(
                gps['lat'], gps['lon'], gps['alt'], gps['heading'],
                gimbal_tracker.pan, gimbal_tracker.tilt,
                (rgb_w // 2, rgb_h // 2)
            )

            if target_location:
                target_lat = target_location.latitude
                target_lon = target_location.longitude
                
                now = time.time()
                if now - last_waypoint_time >= WAYPOINT_INTERVAL:
                    telemetry.set_mode("GUIDED")
                    telemetry.send_waypoint(target_lat, target_lon, gps['alt'])
                    last_waypoint_time = now
                    print(f"[NAV] Waypoint sent: {target_lat:.6f}, {target_lon:.6f}")

        th_display = th_frame.copy()
        if len(th_display.shape) == 2:
            th_display = cv2.cvtColor(th_display, cv2.COLOR_GRAY2BGR)
        
        for det in th_detections:
            cv2.rectangle(th_display, (det.x, det.y), 
                         (det.x + det.width, det.y + det.height), 
                         (0, 0, 255), 2)
            label = f"FIRE {det.max_temp:.0f}C"
            cv2.putText(th_display, label, (det.x, det.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        rgb_display = rgb_frame.copy()
        for i, det in enumerate(rgb_detections):
            if i in confirmed_rgb_indices:
                color = (0, 255, 0)
                label = "Fire Detected"
            else:
                color = (0, 165, 255)
                label = "False Positive"
            
            cv2.rectangle(rgb_display, (det.x, det.y), 
                         (det.x + det.width, det.y + det.height), 
                         color, 2)
            cv2.putText(rgb_display, label, (det.x, det.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        gimbal_tracker.draw_overlay(rgb_display)

        th_resized = cv2.resize(th_display, (int(th_w * rgb_h / th_h), rgb_h))
        combined = np.hstack([rgb_display, th_resized])

        if len(fused) > 0:
            h, w = combined.shape[:2]
            overlay = combined.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 150), -1)
            cv2.addWeighted(overlay, 0.6, combined, 0.4, 0, combined)
            
            best = max(fused, key=lambda d: d.confidence)
            text = f"FIRE DETECTED - {best.thermal_temp:.0f}C"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(combined, text, (text_x, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        h, w = combined.shape[:2]
        if w > MAX_WIDTH:
            scale = MAX_WIDTH / w
            combined = cv2.resize(combined, (MAX_WIDTH, int(h * scale)))

        bar_height = 40
        h, w = combined.shape[:2]
        telemetry_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
        telemetry_bar[:] = (40, 40, 40)
        
        gps = telemetry.gps
        telem_text = f"LAT: {gps['lat']:.6f}  LON: {gps['lon']:.6f}  ALT: {gps['alt']:.1f}m  GS: {gps['groundspeed']:.1f}m/s  HDG: {gps['heading']:.0f}"
        
        if target_lat and target_lon:
            telem_text += f"  | TARGET WAYPOINT: {target_lat:.5f}, {target_lon:.5f}"
        
        cv2.putText(telemetry_bar, telem_text, (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        combined = np.vstack([combined, telemetry_bar])

        cv2.imshow(WINDOW, combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rgb_cam.stop()
    th_cam.stop()
    telemetry.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
