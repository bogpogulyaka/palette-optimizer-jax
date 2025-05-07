from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike
from tqdm import tqdm

from image_processing.base_image_processor import BaseImageProcessor
from palette_generator import PaletteGenerator
from utils.color_operations import ColorOperations
from utils.image_preparation import ImagePreparation
from utils.utils import make_jax_rng_key, chunks
from video_utils.base_video_reader import BaseVideoReader
from video_utils.base_video_writer import BaseVideoWriter, VideoFrameData


class VideoConverter:
    def __init__(self, video_reader: BaseVideoReader, video_writer: BaseVideoWriter, image_processor: BaseImageProcessor, batch_size=200):
        self.video_reader = video_reader
        self.video_writer = video_writer
        self.palette_generator = PaletteGenerator()
        self.image_processor = image_processor

        self.batch_size = batch_size

    def convert(self, sort_palette: bool = True):
        while self._convert_batch(sort_palette):
            pass

    def _convert_batch(self, sort_palette: bool = True):
        frames = self.video_reader.read(self.batch_size)
        n_frames = len(frames)

        if (n_frames == 0):
            self.video_reader.close()
            self.video_writer.close()
            return False

        loss_ratios = np.full((n_frames,), 0.9)
        palettes = self.palette_generator.generate_palettes(frames, loss_ratios)

        if sort_palette:
            palettes = jax.vmap(self._sort_palette)(palettes)

        keys = jax.random.split(make_jax_rng_key(), n_frames)

        # process frames
        for i, (frame, palette, key) in tqdm(enumerate(zip(frames, palettes, keys))):
            image = ImagePreparation.convert_numpy_image_to_jax(frame)
            output_image, color_ids, _ = self.image_processor.process_image(image, palette, key, add_shift=False, return_error=False)

            output_image, color_ids, palette = ImagePreparation.convert_jax_image_to_numpy(output_image), color_ids.__array__().astype(np.uint8), ImagePreparation.convert_jax_image_to_numpy(palette)
            self.video_writer.write(VideoFrameData(output_image, color_ids, palette))

        return True

    @partial(jit, static_argnums=(0,))
    def _sort_palette(self, palette: ArrayLike) -> Array:
        palette_weights = ColorOperations.get_color_weight(palette)
        indices = jnp.argsort(palette_weights)
        return palette[indices]
