import cv2
import numpy as np

from video_utils.base_video_reader import BaseVideoReader
from imutils.video import FileVideoStream


class ImutilsVideoReader(BaseVideoReader):
    def __init__(self, path: str, target_pixel_count=300000, target_framerate=30, block_size=(1, 1)):
        self.cap = cv2.VideoCapture(path)
        framerate = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.release()

        self.vs = FileVideoStream(path).start()

        super().__init__(target_pixel_count, framerate, target_framerate, block_size)

    def close(self):
        self.vs.stop()
        self.vs = None

    def _read_frame(self) -> np.ndarray | None:
        if self.vs is None or not self.vs.more():
            return None

        frame = self.vs.read()
        return frame
