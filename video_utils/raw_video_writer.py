import numpy as np

from video_utils.base_video_writer import BaseVideoWriter, VideoFrameData


class RawVideoWriter(BaseVideoWriter):
    def __init__(self, output_dir: str, target_framerate=30, frames_per_chunk=200):
        super().__init__(target_framerate)

        self.frames_per_chunk = frames_per_chunk
        self.output_dir = output_dir

    def close(self):
        # close file
        pass

    def _write_frame(self, frame: VideoFrameData):
        color_ids = frame.color_ids
        palette = frame.palette.flatten()

        # append bytes to end of file
