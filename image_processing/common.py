from functools import partial

import jax
from jax import jit
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


@partial(jit, static_argnames=('n_iter',))
def get_palette_distribution_algo(palette: ArrayLike, target_color: ArrayLike, n_iter=16, error_mult=0.8) -> Array:
    color_distribution = jnp.zeros(palette.shape[0], dtype=jnp.float32)
    err = jnp.zeros_like(target_color, dtype=jnp.float32)

    for t in range(n_iter):
        current_target = (target_color + err * error_mult).clip(0, 1)

        # Compute penalties for all palette colors
        distances = jnp.sum((palette - current_target) ** 2, axis=-1)
        color_probs = distances == jnp.min(distances)

        chosen_color = jnp.sum(palette * color_probs[:, None], axis=0)
        color_distribution += color_probs

        # Update error
        err += target_color - chosen_color

    return color_distribution / jnp.sum(color_distribution)


@jit
def sample_from_palette_distribution(palette: ArrayLike, distribution: ArrayLike, noise: ArrayLike) -> (Array, Array):
    color_cdf = jnp.cumsum(distribution)
    color_id = jnp.searchsorted(color_cdf, noise, method='compare_all').squeeze()
    color = palette[color_id]

    return color, color_id


@jit
def get_color_representation_error(target_color: ArrayLike, palette: ArrayLike, distribution: ArrayLike) -> (Array, Array):
    average_color = (palette * distribution).sum(-1)

    # Difference with target color
    error = ((target_color - average_color) ** 2).sum()

    # MSE of candidate colors
    distances = ((palette - average_color) ** 2).sum(-1) * distribution
    mse = distances.mean()

    return error, mse
