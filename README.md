# Autonomous Fixed Wing Surveillance Vehicle (AFWSV)

An autonomous fire detection and response system for fixed-wing UAVs. The system uses dual-camera sensor fusion (RGB + thermal) to detect and confirm fire locations, then autonomously navigates the aircraft to orbit around the detected fire.

![Image](https://github.com/user-attachments/assets/bcbb0585-cda4-4f7a-bffc-758166a4fa7d)

## Overview

This project implements a complete autonomous fire surveillance pipeline:

1. **Fire Detection**: Combines RGB camera (color-based detection) with thermal camera (temperature thresholding). Both cameras detect fire independently, and the system uses **sensor fusion** to match detections by comparing normalized pixel coordinates. Only fires confirmed by both sensors are considered valid, eliminating false positives.

2. **Gimbal Tracking**: When a confirmed fire is detected, the gimbal automatically tracks and centers the target in the camera frame.

![Image](https://github.com/user-attachments/assets/46706579-ba00-4596-9e35-8a2fe786f08b)

3. **Target Geolocation**: Once the target is centered, the system calculates the GPS coordinates of the fire using **raycasting**. A virtual ray is projected from the aircraft position, through the camera at the gimbal's current pan/tilt angles, down to the ground plane. The intersection point gives the target's geographic coordinates.

4. **Autonomous Navigation**: The aircraft autonomously navigates toward the fire location. When within the desired distance, it transitions to loiter mode and orbits around the target.

## Target Geolocation (Raycasting)


![Image](https://github.com/user-attachments/assets/137938c5-6806-49b2-beb4-05b093e14726)

The system calculates the fire's ground position using the aircraft's telemetry and gimbal orientation. The core calculation is:

### Ground Distance Calculation

```
ground_distance = altitude / tan(gimbal_tilt + vertical_offset)
```

Where:
- `altitude` is the aircraft's altitude above ground (meters)
- `gimbal_tilt` is the pitch angle of the gimbal (radians)
- `vertical_offset` is the pixel offset from frame center, converted to angle

### Bearing Calculation

```
bearing = heading - (gimbal_pan + horizontal_offset)
```

Where:
- `heading` is the aircraft's compass heading (radians)
- `gimbal_pan` is the yaw angle of the gimbal (radians)
- `horizontal_offset` is the pixel offset from frame center, converted to angle

### GPS Coordinate Projection

The target coordinates are calculated using the Haversine formula inverse:

```
angular_distance = ground_distance / Earth_Radius

target_lat = arcsin(sin(lat) * cos(angular_distance) +
                    cos(lat) * sin(angular_distance) * cos(bearing))

target_lon = lon + arctan2(sin(bearing) * sin(angular_distance) * cos(lat),
                           cos(angular_distance) - sin(lat) * sin(target_lat))
```

This projects a point from the aircraft's position along the calculated bearing at the computed ground distance.

## Dependencies

- OpenCV (cv2)
- NumPy
- PyMAVLink
- Gazebo Transport (gz.transport)

The system will:
1. Connect to MAVLink on `udp:127.0.0.1:14550`
2. Subscribe to RGB and thermal camera topics
3. Display a combined view with detection overlays
4. Automatically control the aircraft when fire is confirmed

## Display Information

The interface shows:
- RGB camera feed with detection boxes (green = confirmed, orange = false positive)
- Thermal camera feed with temperature readings
- Gimbal status (LOCKING countdown, LOCKED indicator)
- Telemetry bar with aircraft position, speed, distance to target, and ROI coordinates

## License

This project is for research and educational purposes.