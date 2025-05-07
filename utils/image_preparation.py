from functools import partial

import cv2
import jax
import numpy as np
from jax import jit

from utils.image_resampler import ImageResampler
from utils.jax_kmeans import JaxKMeans
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


class ImagePreparation:
    @staticmethod
    def convert_numpy_image_to_jax(image: np.ndarray) -> Array:
        return jnp.asarray(image, dtype=np.float32) / 255

    @staticmethod
    def convert_jax_image_to_numpy(image: ArrayLike) -> np.ndarray:
        return (image.clip(0, 1) * 255).__array__().astype(np.uint8)

    @staticmethod
    def downscale_image(image: np.ndarray, target_pixel_count=1000000, interpolation=cv2.INTER_AREA, block_size: (int, int) = (1, 1)) -> np.ndarray:
        image_resampler = ImageResampler()
        image_size = image_resampler.get_size_for_pixel_count(image.shape, target_pixel_count, block_size)
        return cv2.resize(image, (image_size[1], image_size[0]), interpolation=interpolation)

    @staticmethod
    @partial(jit, static_argnames=('n_color_clusters', 'add_noise'))
    def get_image_pixels_color_weights(pixels: ArrayLike, n_color_clusters: int, rng_key: ArrayLike, add_noise=True) -> (Array, Array):
        if add_noise:
            noise_magnitude = 0.005
            noise = jax.random.uniform(rng_key, pixels.shape)
            pixels = pixels * (1 - noise_magnitude) + noise * noise_magnitude

        kmeans = JaxKMeans(n_clusters=n_color_clusters, max_iter=50)
        colors, color_ids, _ = kmeans.fit(pixels, rng_key)

        weights = jnp.bincount(color_ids, length=n_color_clusters).astype(jnp.float32)
        # weights = weights ** 0.3
        weights = weights / weights.mean()

        return colors, weights
