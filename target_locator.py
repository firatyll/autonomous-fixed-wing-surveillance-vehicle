import math
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TargetLocation:
    latitude: float
    longitude: float
    distance: float
    bearing: float


class TargetLocator:
    """Calculates target GPS location from UAV telemetry and camera data."""
    
    # Camera and world constants
    VERTICAL_FOV = 0.9273
    HORIZONTAL_FOV = 1.57
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    EARTH_RADIUS = 6371000.0
    
    def __init__(self, frame_width: int = 1280, frame_height: int = 720,
                 horizontal_fov: float = 1.57, vertical_fov: float = 0.9273):
        """Initialize with camera parameters.
        
        Args:
            frame_width: Camera frame width in pixels.
            frame_height: Camera frame height in pixels.
            horizontal_fov: Horizontal field of view in radians.
            vertical_fov: Vertical field of view in radians.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
    
    def locate_target(self, uav_lat: float, uav_lon: float, uav_alt: float,
                      uav_heading: float, gimbal_pan: float, gimbal_tilt: float,
                      target_center: Tuple[int, int]) -> Optional[TargetLocation]:
        """Calculate target GPS location from UAV position and gimbal angles.
        
        Args:
            uav_lat: UAV latitude in degrees.
            uav_lon: UAV longitude in degrees.
            uav_alt: UAV altitude in meters.
            uav_heading: UAV heading in degrees.
            gimbal_pan: Gimbal pan/yaw angle in radians.
            gimbal_tilt: Gimbal tilt/pitch angle in radians.
            target_center: Target pixel coordinates (x, y).
            
        Returns:
            TargetLocation with calculated GPS coordinates, or None on error.
        """
        try:
            target_x, target_y = target_center
            
            center_x = self.frame_width / 2.0
            center_y = self.frame_height / 2.0
            offset_x = (target_x - center_x) / center_x
            offset_y = (target_y - center_y) / center_y
            
            angle_offset_pan = -offset_x * (self.horizontal_fov / 2.0)
            angle_offset_tilt = offset_y * (self.vertical_fov / 2.0)
            
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
            angular_dist = ground_distance / self.EARTH_RADIUS
            
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
    
    def localize_from_telemetry(self, gps: Dict[str, Any], gimbal_pan: float,
                                 gimbal_tilt: float, target_center: Tuple[int, int]) -> Optional[TargetLocation]:
        """Convenience method to locate target from GPS telemetry dict.
        
        Args:
            gps: Dict with 'lat', 'lon', 'alt', 'heading' keys.
            gimbal_pan: Gimbal pan/yaw angle in radians.
            gimbal_tilt: Gimbal tilt/pitch angle in radians.
            target_center: Target pixel coordinates (x, y).
            
        Returns:
            TargetLocation with calculated GPS coordinates, or None on error.
        """
        return self.locate_target(
            gps['lat'], gps['lon'], gps['alt'], gps['heading'],
            gimbal_pan, gimbal_tilt, target_center
        )