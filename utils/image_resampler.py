import math

import cv2
import numpy as np
import typing as tp

from scipy.stats import qmc


class ImageResampler:
    def __init__(self):
        pass

    def get_size_for_pixel_count(self, original_size: tp.Tuple[int, int], pixel_count: int, block_size: (int, int) = (1, 1)) -> tp.Tuple[int, int]:
        ratio = original_size[0] / original_size[1]
        side = math.sqrt(pixel_count * ratio)
        return round(side / block_size[0]) * block_size[0], round(side / ratio / block_size[1]) * block_size[1]

    def downsample_nearest_cv2(self, image: np.ndarray, size: tp.Tuple[int, int]) -> np.ndarray:
        return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_NEAREST_EXACT)
