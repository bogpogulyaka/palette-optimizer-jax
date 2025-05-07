import cv2
import torch
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from torch2jax import j2t, t2j

from models.palette_generation_model_poly import PaletteGenerationModelPoly, evaluate_polynomial
from utils.image_preparation import ImagePreparation
from utils.utils import ROOT_DIR, make_jax_rng_key

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200, threshold=1000000)


class PaletteGenerator:
    def __init__(self, n_color_clusters=128, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_color_clusters = n_color_clusters
        self.model = self.load_model(model_path)

    def load_model(self, model_path=None):
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_64b_16col__43.pth'
        # path = f'{ROOT_DIR}/checkpoints/gen_transformer_128d_6l_8h_1m_64b_16col__2.pth'
        # path = f'{ROOT_DIR}/checkpoints/gen_transformer_128d_6l_8h_50k_64b_16col__3.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_64b_16col_128clst_stack1_varratio_1.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_13m_64b_16col_128clst_stack1_varratio_2_epoch5.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_13m_64b_16col_128clst_stack1_varratio_7_epoch5.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_64b_16col_128clst_stack1_varratio_8_epoch9.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_32b_16col_256clst_stack2_varratio_10_epoch5.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_50k_64b_16col_128clst_stack1_varratio_bndloss_1_epoch3.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_50k_64b_16col_128clst_stack1_varratio_bndloss_3_epoch3.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_64b_16col_128clst_stack1_varratio_bndloss_5_epoch1.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_32b_16col_128clst_stack1_varratio_bndloss_7_epoch1.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_32b_16col_192clst_stack1_varratio_bndloss_8_epoch10.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_32b_16col_128clst_stack1_poly3_tgtvar_6_epoch10.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_32b_16col_128_poly3_restr_1_epoch2.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_32b_16col_128_poly3_restr_2_epoch1.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_32b_16col_128_poly3_lms_2_epoch1.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_32b_16col_128_poly3_lms_restr_1_epoch1.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_32b_16col_128_poly3_regs_1_epoch1.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_32b_16col_128_poly3_denoise_1_epoch1.pth'
        path = f'checkpoints/gen_transformer_256d_8l_8h_1m_32b_16col_128_poly3_distr_1_epoch10.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_32b_8col_128_8colors_poly3_distr_2_epoch1.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_128b_16col_128_poly3_distr_algoloss_1_epoch1.pth'
        # path = f'checkpoints/gen_transformer_256d_8l_8h_1m_64b_16col_128_poly3_distr_algoloss_8_epoch1.pth'

        if model_path is not None:
            path = model_path

        checkpoint = torch.load(f'{ROOT_DIR}/{path}')

        model = PaletteGenerationModelPoly(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])

        model.to(self.device)
        model.eval()

        return model

    def generate_palettes(self, images: list[np.ndarray], loss_ratios: np.ndarray) -> Array:
        images_pixels = []
        for image in images:
            image = ImagePreparation.downscale_image(image, 5000, cv2.INTER_NEAREST)
            pixels = jax.lax.collapse(ImagePreparation.convert_numpy_image_to_jax(image), 0, 2)
            images_pixels.append(pixels)
        images_pixels = jnp.stack(images_pixels)

        rng_key = make_jax_rng_key()

        colors, weights = jax.vmap(ImagePreparation.get_image_pixels_color_weights, in_axes=(0, None, None))(images_pixels, self.n_color_clusters, rng_key)
        colors, weights = j2t(colors), j2t(weights)
        # _colors, _weights = torch.from_numpy(colors.__array__()).to(self.device), torch.from_numpy(weights.__array__()).to(self.device)

        loss_ratios = torch.tensor(loss_ratios, device=self.device)

        with torch.cuda.amp.autocast():
            # palettes = self.model(colors, weights, loss_ratios)
            palettes, _ = self.model(colors, weights, loss_ratios)
            palettes = palettes.clip(0, 1)

        torch.cuda.synchronize()
        palettes = t2j(palettes)
        # palettes = jnp.array(palettes.numpy(force=True))

        return palettes
