import os

import numpy as np
from PIL import Image

from video_utils.base_video_writer import BaseVideoWriter, VideoFrameData


class DummyVideoWriter(BaseVideoWriter):
    def __init__(self):
        super().__init__(30)

    def close(self):
        pass

    def _write_frame(self, frame: VideoFrameData):
        pass
