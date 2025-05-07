import jax
import torch
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

class ColorOperations:
    # color_mult = jnp.array([0.399, 0.487, 0.114])
    color_mult = jnp.array([0.299, 0.587, 0.114])

    # OkLab matrices
    OkLabM1 = jnp.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ])
    OkLabM2 = jnp.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ])
    OkLabM1_inv = jnp.array([
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480]
    ])
    OkLabM2_inv = jnp.array([
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010]
    ])

    @staticmethod
    def get_color_weight(color: ArrayLike) -> Array:
        return (color * ColorOperations.color_mult).sum(-1)

    @staticmethod
    def compare_colors(c1: ArrayLike, c2: ArrayLike) -> Array:
        c1 = c1 ** (1 / 2.2)
        c2 = c2 ** (1 / 2.2)
        diff = c1 - c2
        lumadiff = jnp.sum(diff * ColorOperations.color_mult, axis=-1)
        return jnp.sum(diff * diff * ColorOperations.color_mult, axis=-1) * 0.75 + lumadiff * lumadiff

    @staticmethod
    def linear_to_gamma_space(color: ArrayLike) -> Array:
        return color ** (1 / 2.2)

    @staticmethod
    def gamma_to_linear_space(color: ArrayLike) -> Array:
        return color ** 2.2

    @staticmethod
    def linear_srgb_to_oklab(color: ArrayLike) -> Array:
        lms = ColorOperations.OkLabM1 @ color[:, None]
        lms_cbrt = lms.sign() * lms.abs() ** (1/3)
        oklab = (ColorOperations.OkLabM2 @ lms_cbrt).squeeze(-1)

        return oklab

    @staticmethod
    def oklab_to_linear_srgb(color: ArrayLike) -> Array:
        lms_cbrt = ColorOperations.OkLabM1_inv @ color[:, None]
        lms = lms_cbrt ** 3
        linear_srgb = (ColorOperations.OkLabM2_inv @ lms).squeeze(-1)

        return linear_srgb

    @staticmethod
    def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_h = torch.empty_like(rgb[:, 0:1])
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.
        hsv_h /= 6.
        hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


