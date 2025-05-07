from functools import partial

import jax
from jax import numpy as jnp, jit
from jax import Array
from jax.typing import ArrayLike

from image_processing.base_image_processor import BaseImageProcessor
from image_processing.common import get_palette_distribution_algo
from image_processing.enums import NoiseType


class AlgoImageProcessor(BaseImageProcessor):
    def __init__(self, noise_type: NoiseType = NoiseType.BLUE):
        super().__init__(noise_type)

    @partial(jit, static_argnames=('self',))
    def _get_pixels_color_distributions(self, pixels: ArrayLike, palette: ArrayLike) -> Array:
        return jax.vmap(get_palette_distribution_algo, in_axes=(None, 0))(palette, pixels)
