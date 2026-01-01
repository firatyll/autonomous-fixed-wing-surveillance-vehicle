import threading
from typing import Optional, Dict, Any
from pymavlink import mavutil


class Telemetry:
    """Unified MAVLink telemetry and navigation controller."""
    
    def __init__(self, connection_string: str = "udp:127.0.0.1:14550"):
        self._connection_string = connection_string
        self._mav: Optional[mavutil.mavlink_connection] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        self._gps_data: Dict[str, Any] = {
            "lat": 0.0,
            "lon": 0.0,
            "alt": 0.0,
            "groundspeed": 0.0,
            "heading": 0.0,
            "satellites": 0,
            "fix_type": 0,
        }
        self._lock = threading.Lock()
    
    def set_mode(self, mode_name: str) -> None:
        """Set the flight mode (e.g., 'GUIDED', 'AUTO')."""
        with self._lock:
            if self._mav is None:
                return
            mode_id = self._mav.mode_mapping().get(mode_name.upper())
            if mode_id is None:
                return
            self._mav.mav.set_mode_send(
                self._mav.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )
    
    def send_waypoint(self, lat: float, lon: float, alt: float) -> None:
        """Send a waypoint command."""
        with self._lock:
            if self._mav is None:
                return
            self._mav.mav.mission_item_int_send(
                self._mav.target_system, self._mav.target_component, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 2, 1, 0, 0, 0, 0,
                int(lat * 1e7), int(lon * 1e7), alt
            )
    
    def set_roi(self, lat: float, lon: float, alt: float = 0) -> None:
        """Set Region of Interest for gimbal/camera pointing."""
        with self._lock:
            if self._mav is None:
                return
            self._mav.mav.command_long_send(
                self._mav.target_system, self._mav.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_ROI_LOCATION, 0, 0, 0, 0, 0,
                lat, lon, alt
            )
    
    def set_circle_radius(self, radius: float) -> None:
        """Set the loiter/circle radius in meters."""
        with self._lock:
            if self._mav is None:
                return
            self._mav.mav.param_set_send(
                self._mav.target_system, self._mav.target_component,
                b'WP_LOITER_RAD', float(radius),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
    
    def send_loiter(self, lat: float, lon: float, alt: float) -> None:
        """Send a loiter/orbit command."""
        with self._lock:
            if self._mav is None:
                return
            self._mav.mav.mission_item_int_send(
                self._mav.target_system, self._mav.target_component, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM, 2, 1, 0, 0, 0, 0,
                int(lat * 1e7), int(lon * 1e7), alt
            )
    
    def set_speed(self, speed: float) -> None:
        """Set the airspeed in m/s."""
        with self._lock:
            if self._mav is None:
                return
            self._mav.mav.command_long_send(
                self._mav.target_system, self._mav.target_component,
                mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0, 1, speed, -1, 0, 0, 0, 0
            )
    
    @property
    def connection(self):
        """Returns the MAVLink connection object."""
        return self._mav
    
    @property
    def gps(self) -> Dict[str, Any]:
        """Returns the latest GPS data (thread-safe)."""
        with self._lock:
            return self._gps_data.copy()
    
    def _update_loop(self) -> None:
        """Background thread loop for receiving MAVLink messages."""
        while self._running:
            try:
                msg = self._mav.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    continue
                
                msg_type = msg.get_type()
                
                if msg_type == "GLOBAL_POSITION_INT":
                    with self._lock:
                        self._gps_data["lat"] = msg.lat / 1e7
                        self._gps_data["lon"] = msg.lon / 1e7
                        self._gps_data["alt"] = msg.relative_alt / 1000.0
                        self._gps_data["heading"] = msg.hdg / 100.0
                
                elif msg_type == "VFR_HUD":
                    with self._lock:
                        self._gps_data["groundspeed"] = msg.groundspeed
                
                elif msg_type == "GPS_RAW_INT":
                    with self._lock:
                        self._gps_data["satellites"] = msg.satellites_visible
                        self._gps_data["fix_type"] = msg.fix_type
                        
            except Exception as e:
                print(f"Telemetry error: {e}")
    
    def start(self) -> bool:
        """Start the telemetry connection and background thread."""
        try:
            print(f"Connecting to {self._connection_string}...")
            self._mav = mavutil.mavlink_connection(self._connection_string)
            self._mav.wait_heartbeat(timeout=10)
            print("MAVLink heartbeat received!")
            
            self._running = True
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the telemetry thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("Telemetry stopped.")
