"""
Telemetry Module
Handles MAVLink GPS data retrieval in a separate thread.
"""

import threading
from typing import Optional, Dict, Any
from pymavlink import mavutil


class Telemetry:
    """Receives GPS telemetry data via MAVLink in a background thread."""
    
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
