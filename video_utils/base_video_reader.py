from abc import ABC, abstractmethod

import cv2
import numpy as np

from utils.image_preparation import ImagePreparation


class BaseVideoReader(ABC):
    def __init__(self, target_pixel_count=300000, input_framerate=30, target_framerate=30, block_size=(1, 1)):
        self.target_pixel_count = target_pixel_count
        self.input_framerate = input_framerate
        self.target_framerate = target_framerate
        self.block_size = block_size
        self.frame_debt = 0

    @abstractmethod
    def close(self):
        pass

    def read(self, count: int) -> list[np.ndarray]:
        frames = []

        while len(frames) < count:
            frame = self._read_frame()
            if frame is None:
                break

            self.frame_debt += (self.input_framerate - self.target_framerate) / self.input_framerate
            prepared_frame = None

            if self.frame_debt >= 1:  # skip
                self.frame_debt -= 1
                continue

            while self.frame_debt <= -1:  # repeat
                self.frame_debt += 1
                if prepared_frame is None:
                    prepared_frame = self._prepare_frame(frame)
                frames.append(prepared_frame)

            if prepared_frame is None:
                prepared_frame = self._prepare_frame(frame)
            frames.append(prepared_frame)

        return frames

    @abstractmethod
    def _read_frame(self) -> np.ndarray | None:
        pass

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = ImagePreparation.downscale_image(frame, self.target_pixel_count, block_size=self.block_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
