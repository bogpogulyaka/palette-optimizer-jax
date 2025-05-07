from abc import ABC, abstractmethod
from enum import Enum
from functools import partial

import cv2
import jax
import torch
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax import Array
from jax.typing import ArrayLike

from image_processing.common import sample_from_palette_distribution, get_color_representation_error
from image_processing.enums import NoiseType
from utils.color_operations import ColorOperations
from utils.utils import ROOT_DIR, reweight_distribution


class BaseImageProcessor(ABC):
    def __init__(self, noise_type: NoiseType):
        self.noise_texture = self._load_noise_texture(noise_type)

    # @partial(jit, static_argnames=('self', 'return_image', 'return_error', 'add_shift'))
    def process_image(self, image: ArrayLike, palette: ArrayLike, rng_key, temperature=1.0, return_image=True, return_error=False, add_shift=False) -> (Array, Array, Array):
        height, width = image.shape[:2]
        # flatten image
        image = image.reshape((height * width, -1))

        distributions = self._get_pixels_color_distributions(image, palette)
        distributions = reweight_distribution(distributions, temperature)
        noise = self._get_tiled_noise(width, height, rng_key, add_shift)

        if return_image:
            output_image, color_ids = self._get_pixels_from_color_distribution(palette, distributions, noise)
            # reshape
            output_image = output_image.reshape((height, width, -1))
            color_ids = color_ids.reshape((height, width))

        if return_error:
            mean_err, var_err = self._get_pixels_representation_error(palette, distributions, noise)
            # reshape
            err_image = jnp.concatenate([mean_err.unsqueeze(-1), mean_err.unsqueeze(-1), jnp.zeros((*mean_err.shape, 1))], axis=-1)
            err_image = err_image.reshape((height, width, -1))

        return (return_image and (output_image, color_ids) or (None, None)) + (return_error and (err_image,) or (None,))

    @abstractmethod
    def _get_pixels_color_distributions(self, pixels: ArrayLike, palette: ArrayLike) -> Array:
        pass

    def _load_noise_texture(self, noise_type: NoiseType) -> Array:
        paths = {
            NoiseType.BAYER_2: 'bayer2.png',
            NoiseType.BAYER_4: 'bayer4.png',
            NoiseType.BAYER_8: 'bayer8.png',
            NoiseType.BAYER_16: 'bayer16.png',
            NoiseType.BLUE: 'blue-noise.png',
        }
        noise_path = f'{ROOT_DIR}/data/noise_textures/{paths[noise_type]}'
        # noise_path = f'{ROOT_DIR}/data/noise_textures/blue-noise.png'

        texture = cv2.imread(noise_path)[:, :, 0].astype(np.float32) / 256

        return jnp.array(texture).clip(0.001, 0.999)

    @partial(jit, static_argnames=('self', 'width', 'height', 'add_shift'))
    def _get_tiled_noise(self, width: int, height: int, rng_key: ArrayLike, add_shift=False) -> Array:
        size_x, size_y = self.noise_texture.shape
        xx, yy = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing='ij')

        # shift_x, shift_y = random.randint(0, size_x - 1), random.randint(0, size_y - 1)
        shift = jax.random.randint(rng_key, (2,), 0, jnp.array([size_x, size_y]))
        shift *= int(add_shift)

        return self.noise_texture[(xx + shift[0]) % size_x, (yy + shift[1]) % size_y].reshape(-1)

    @partial(jit, static_argnames=('self',))
    def _get_pixels_from_color_distribution(self, palette: ArrayLike, distributions: ArrayLike, noise: ArrayLike) -> (Array, Array):
        image, color_ids = jax.vmap(sample_from_palette_distribution, in_axes=(None, 0, 0))(palette, distributions, noise)
        return image, color_ids

    @partial(jit, static_argnames=('self',))
    def _get_pixels_representation_error(self, target_pixels: ArrayLike, palette: ArrayLike, distributions: ArrayLike) -> (Array, Array):
        target_pixels = self._rgb_to_color_space(target_pixels)
        palette = self._rgb_to_color_space(palette)

        mean_err, var_err = jax.vmap(get_color_representation_error)(target_pixels, palette, distributions)
        return mean_err, var_err

    # def append_palette_to_image(self, image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    #     image_height, image_width = image.shape[0], image.shape[1]
    #     block_height = 20
    #     palette_block = np.zeros((block_height, image_width, 3), dtype=np.float32)
    #
    #     for i, color in enumerate(palette):
    #         palette_block[:, i * image_width // len(palette):(i + 1) * image_width // len(palette), :] = color
    #
    #     return np.concatenate((image, palette_block), axis=0)
    #
    # def append_palette_history_to_image(self, image: np.ndarray, palette_history: list) -> np.ndarray:
    #     image_height, image_width = image.shape[0], image.shape[1]
    #     history_slice_height = 2
    #     palette_block = np.zeros((history_slice_height * len(palette_history), image_width, 3), dtype=np.float32)
    #
    #     for i, palette in enumerate(palette_history):
    #         for j, color in enumerate(palette):
    #             palette_block[i * history_slice_height: (i + 1) * history_slice_height, j * image_width // len(palette):(j + 1) * image_width // len(palette), :] = color
    #
    #     return np.concatenate((image, palette_block), axis=0)

    @partial(jit, static_argnames=('self',))
    def _rgb_to_color_space(self, c: ArrayLike) -> Array:
        c = ColorOperations.gamma_to_linear_space(c)
        c = ColorOperations.linear_srgb_to_oklab(c.clip(0.001, 1))
        return c

    @partial(jit, static_argnames=('self',))
    def _color_space_to_rgb(self, c: ArrayLike) -> Array:
        c = ColorOperations.oklab_to_linear_srgb(c)
        c = ColorOperations.linear_to_gamma_space(c)
        return c
