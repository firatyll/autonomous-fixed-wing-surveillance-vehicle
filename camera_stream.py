import cv2
import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node
from typing import Callable, Optional


class CameraStream:
    """Subscribes to Gazebo camera topic and processes frames."""
    
    def __init__(self, topic: str = "/camera", window_name: str = "Mini Talon Camera"):
        self._topic = topic
        self._window_name = window_name
        self._node: Optional[Node] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_callback: Optional[Callable[[np.ndarray], None]] = None
        
    @property
    def frame(self) -> Optional[np.ndarray]:
        """Returns the latest frame (BGR format)."""
        return self._frame
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set a callback to be called when a new frame arrives."""
        self._frame_callback = callback
    
    def _on_image(self, msg: Image) -> None:
        """Internal callback for processing incoming camera images."""
        img_data = np.frombuffer(msg.data, dtype=np.uint8)
        
        total_pixels = msg.height * msg.width
        channels = len(img_data) // total_pixels
        
        if channels == 3:   
            img = img_data.reshape((msg.height, msg.width, 3))
            self._frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif channels == 1:
            img = img_data.reshape((msg.height, msg.width))
            self._frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = img_data.reshape((msg.height, msg.width, channels))
            self._frame = img
        
        if self._frame_callback:
            self._frame_callback(self._frame)
    
    def start(self) -> bool:
        """Start subscribing to the camera topic."""
        self._node = Node()
        print(f"Subscribing to {self._topic}...")
        return self._node.subscribe(Image, self._topic, self._on_image)
    
    def show_frame(self) -> None:
        """Display the current frame in an OpenCV window."""
        if self._frame is not None:
            cv2.imshow(self._window_name, self._frame)
            cv2.waitKey(1)
    
    def stop(self) -> None:
        """Stop the camera stream and close windows."""
        print("Closing camera stream...")
        cv2.destroyAllWindows()
