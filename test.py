from utils.color_operations import ColorOperations
from utils.utils import chunks
import jax.numpy as jnp

color = jnp.array([[0, 0.5, 1], [0, 0.5, 1]])
w = ColorOperations.get_color_weight(color)
print(w.shape)
