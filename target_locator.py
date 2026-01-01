import math
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

VERTICAL_FOV = 0.9273
HORIZONTAL_FOV = 1.57
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
EARTH_RADIUS = 6371000.0

@dataclass
class TargetLocation:
    latitude: float
    longitude: float
    distance: float
    bearing: float

def locate_target(uav_lat: float, uav_lon: float, uav_alt: float, 
                  uav_heading: float, gimbal_pan: float, gimbal_tilt: float, 
                  target_center: Tuple[int, int]) -> Optional[TargetLocation]:
    try:
        target_x, target_y = target_center
        
        center_x = FRAME_WIDTH / 2.0
        center_y = FRAME_HEIGHT / 2.0
        offset_x = (target_x - center_x) / center_x
        offset_y = (target_y - center_y) / center_y
        
        angle_offset_pan = -offset_x * (HORIZONTAL_FOV / 2.0)
        angle_offset_tilt = offset_y * (VERTICAL_FOV / 2.0)
        
        total_pan = gimbal_pan + angle_offset_pan
        total_tilt = gimbal_tilt + angle_offset_tilt
        
        ground_distance = uav_alt / math.tan(total_tilt)
        
        if ground_distance < 0 or ground_distance > 10000:
            return None
        
        heading_rad = math.radians(uav_heading)
        bearing_rad = heading_rad - total_pan
        bearing_rad = bearing_rad % (2 * math.pi)
        bearing_deg = math.degrees(bearing_rad)
        
        lat_rad = math.radians(uav_lat)
        lon_rad = math.radians(uav_lon)
        angular_dist = ground_distance / EARTH_RADIUS
        
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(angular_dist) +
            math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bearing_rad)
        )
        
        new_lon_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_dist) * math.cos(lat_rad),
            math.cos(angular_dist) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        
        return TargetLocation(
            latitude=math.degrees(new_lat_rad),
            longitude=math.degrees(new_lon_rad),
            distance=ground_distance,
            bearing=bearing_deg
        )
        
    except Exception as e:
        print(f"[LOCATOR] Error: {e}")
        return None


def localize_from_telemetry(gps: Dict[str, Any], gimbal_pan: float, 
                             gimbal_tilt: float, target_center: Tuple[int, int]) -> Optional[TargetLocation]:
    return locate_target(
        gps['lat'], gps['lon'], gps['alt'], gps['heading'],
        gimbal_pan, gimbal_tilt, target_center
    )