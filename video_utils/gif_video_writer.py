import os

import numpy as np
from PIL import Image

from video_utils.base_video_writer import BaseVideoWriter, VideoFrameData


class GIFVideoWriter(BaseVideoWriter):
    def __init__(self, output_dir: str, target_framerate=30, frames_per_clip=200):
        super().__init__(target_framerate)

        self.frames_per_clip = frames_per_clip
        self.output_dir = output_dir
        self.output_images = []
        self.clip_counter = 1

    def close(self):
        self._save_clip()

    def _write_frame(self, frame: VideoFrameData):
        palette = frame.palette.flatten()
        color_ids = frame.color_ids
        # color_ids = self._decompose_frame(color_ids, (2, 2))

        # output_frame = Image.fromarray(frame.image)
        output_frame = Image.fromarray(color_ids, 'P')
        output_frame.putpalette(palette)

        self.output_images.append(output_frame)
        if (len(self.output_images) >= self.frames_per_clip):
            self._save_clip()

    def _decompose_frame(self, frame: np.ndarray, block_size: (int, int)) -> np.ndarray:
        decomposed_frame = np.empty_like(frame)
        fs0, fs1 = frame.shape[0], frame.shape[1]
        bs0, bs1 = block_size
        ps0, ps1 = fs0 // bs0, fs1 // bs1

        for i in range(bs0):
            for j in range(bs1):
                decomposed_frame[ps0 * i: ps0 * (i + 1), ps1 * j: ps1 * (j + 1)] = frame[i::bs0, j::bs1]

        return decomposed_frame

    def _save_clip(self):
        if len(self.output_images) == 0:
            return

        try:
            os.mkdir(self.output_dir)
        except:
            pass

        duration = 1000 // self.target_framerate

        self.output_images[0].save(
            f"{self.output_dir}/{self.clip_counter}.gif",
            save_all=True,
            append_images=self.output_images[1:],
            optimize=False,
            duration=duration,
            loop=0,
        )
        self.output_images = []

        self.clip_counter += 1
