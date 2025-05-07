import cv2
import numpy as np

from video_utils.base_video_reader import BaseVideoReader


class CV2VideoReader(BaseVideoReader):
    def __init__(self, path: str, target_pixel_count=300000, target_framerate=30, block_size=(1, 1)):
        self.cap = cv2.VideoCapture(path)
        framerate = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.cap.isOpened():
            print("Error: Could not open video.")

        super().__init__(target_pixel_count, framerate, target_framerate, block_size)

    def close(self):
        self.cap.release()
        self.cap = None

    def _read_frame(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None
