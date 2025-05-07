from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class VideoFrameData:
    image: np.ndarray = None
    color_ids: np.ndarray = None
    palette: np.ndarray = None


class BaseVideoWriter(ABC):
    def __init__(self, target_framerate):
        self.target_framerate = target_framerate
        self.frame_debt = 0

    @abstractmethod
    def close(self):
        pass

    def write(self, frame: VideoFrameData, input_framerate: int = None):
        input_framerate = input_framerate or self.target_framerate

        self.frame_debt += (input_framerate - self.target_framerate) / input_framerate

        if self.frame_debt >= 1:  # skip
            self.frame_debt -= 1
            return

        while self.frame_debt <= -1:  # repeat
            self.frame_debt += 1
            self._write_frame(frame)

        self._write_frame(frame)


    @abstractmethod
    def _write_frame(self, frame: VideoFrameData):
        pass
