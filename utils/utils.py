import random
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import jit
from jax import Array
from jax.typing import ArrayLike


def get_project_root() -> Path:
    return Path(__file__).parent.parent


ROOT_DIR = get_project_root()


@jit
def reweight_distribution(distribution: ArrayLike, temperature=1.0) -> Array:
    return jax.nn.softmax(jnp.log(distribution) / temperature)


def make_jax_rng_key():
    return jax.random.key(random.randint(1, 10 ** 9))


def chunks(iterable, chunk_size):
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]
