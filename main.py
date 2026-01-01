import cv2
import numpy as np
import time
from camera_stream import CameraStream
from telemetry import Telemetry
from thermal_detector import ThermalDetector
from fire_detector import FireDetector
from sensor_fusion import SensorFusion
from gimbal_tracker import GimbalTracker
from target_locator import localize_from_telemetry
from navigation_utils import set_mode, send_waypoint, set_roi, set_circle_radius, send_loiter, set_speed

RGB_TOPIC = "/world/runway/model/mini_talon_vtail/link/camera_tilt_link/sensor/camera/image"
THERMAL_TOPIC = "/thermal_camera"
MAV_CONN = "udp:127.0.0.1:14550"
WINDOW = "Mini Talon"
INFO_BAR_H = 80
TH_THRESH = 500.0

ALT = 70.0
SPEED_APP = 13.0
SPEED_ORB = 10.0
DIST_ORB = 130.0
RAD_ORB = 85.0

def draw_ui(frame, gps, target, center, finished, mode):
    h, w = frame.shape[:2]
    bar = np.zeros((INFO_BAR_H, w, 3), dtype=np.uint8) + 40
    
    uav_txt = f"UAV  | LAT: {gps['lat']:.6f}  LON: {gps['lon']:.6f}  ALT: {gps['alt']:.1f}m  HDG: {gps['heading']:.0f}  MODE: {mode}"
    cv2.putText(bar, uav_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    if target:
        c = (0, 255, 0) if center else (0, 0, 255)
        st = "[FINISHED]" if finished else ("[CENTERED]" if center else "[TRACKING]")
        fire_txt = f"FIRE | LAT: {target.latitude:.6f}  LON: {target.longitude:.6f}  DIST: {target.distance:.0f}m  BRG: {target.bearing:.0f} {st}"
        cv2.putText(bar, fire_txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    else:

    status_msg = "MISSION COMPLETE: ORBITING TARGET" if finished else ("AUTONOMOUS APPROACH ACTIVE" if mode == "GUIDED" else "WAITING FOR LOCK...")
    status_color = (0, 255, 0) if finished else ((0, 255, 255) if mode == "GUIDED" else (200, 200, 200))
    cv2.putText(bar, f"STATUS: {status_msg}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    return np.vstack((frame, bar))

def main():
    rgb_cam = CameraStream(RGB_TOPIC)
    th_cam = CameraStream(THERMAL_TOPIC)
    mav = Telemetry(MAV_CONN)
    th_det = ThermalDetector(TH_THRESH)
    rgb_det = FireDetector()
    fusion = SensorFusion(0.15)
    tracker = GimbalTracker()
    
    if not (rgb_cam.start() and th_cam.start() and mav.start()): return
    
    orbiting = False
    autonomous = False
    target_loc = None
    first_lock = False
    last_upd = 0
    
    try:
        while True:
            rgb, th, gps = rgb_cam.frame, th_cam.frame, mav.gps
            if rgb is None or th is None: continue

            rgb_d = rgb_det.detect(rgb)
            th_d = th_det.detect(th)
            fused, conf_indices = fusion.fuse(rgb_d, th_d, rgb.shape[:2][::-1], th.shape[:2][::-1])
            
            tracker.update(fused, rgb.shape[:2])
            
            conn = mav.connection
            if not orbiting and conn:
                now = time.time()
                update = (tracker.is_centered and fused and not first_lock) or \
                         (first_lock and fused and (now - last_upd) >= 5.0)
                
                if update:
                    best = max(fused, key=lambda d: d.confidence)
                    loc = localize_from_telemetry(gps, tracker.current_pan, tracker.current_tilt, best.rgb_center)
                    
                    if loc:
                        target_loc = loc
                        last_upd = now
                        
                        if not first_lock:
                            first_lock = True
                            set_speed(conn, SPEED_APP)
                            set_mode(conn, "GUIDED")
                            autonomous = True
                        
                        if autonomous:
                            send_waypoint(conn, loc.latitude, loc.longitude, ALT)
                        
                        if loc.distance < DIST_ORB:
                            set_speed(conn, SPEED_ORB)
                            set_roi(conn, loc.latitude, loc.longitude, 0)
                            set_circle_radius(conn, RAD_ORB)
                            send_loiter(conn, loc.latitude, loc.longitude, ALT)
                            orbiting = True

            th_vis = th.copy()
            if th_d:
                th_vis = th_det.draw_detections(th_vis, th_d)

            rgb_vis = rgb.copy()
            if rgb_d:
                rgb_vis = rgb_det.draw_detections(rgb_vis, rgb_d, conf_indices)
            
            rgb_vis = tracker.draw_overlay(rgb_vis)

            disp = np.hstack((rgb_vis, th_vis))
            disp = cv2.resize(disp, (0, 0), fx=0.5, fy=0.5)
            
            if fused:
                disp = fusion.draw_warning(disp, fused)

            cv2.imshow(WINDOW, draw_ui(disp, gps, target_loc, tracker.is_centered, orbiting, "AUTO" if autonomous else "MAN"))
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            time.sleep(0.001)

    finally:
        rgb_cam.stop(); th_cam.stop(); mav.stop()

if __name__ == "__main__":
    main()