import os
import random
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from utils.utils import make_jax_rng_key


class JaxKMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.n_iter = max_iter
        self.cluster_centers_ = None
        self.inertia_ = None
        self.labels_ = None

    @partial(jax.jit, static_argnums=(0,))
    def _fit_loop(self, x: ArrayLike, rng_key: ArrayLike) -> (Array, Array, Array):
        indices = jax.random.permutation(rng_key, jnp.arange(x.shape[0]))[:self.n_clusters]
        means = x[indices]

        # eye_mat = jnp.eye(self.n_clusters)

        for _ in range(self.n_iter):
            diff_from_means = means[None, :] - x[:, None]
            dist_to_means = (diff_from_means ** 2).sum(-1)

            means_ids = dist_to_means.argmin(-1)
            # a = eye_mat[means_ids].T
            a = jax.nn.one_hot(means_ids, self.n_clusters).T

            means = a @ x / (a.sum(-1, keepdims=True) + 1e-10)

        return means, means_ids, dist_to_means.min(-1).sum(-1)

    @partial(jax.jit, static_argnums=(0,))
    def fit(self, x: ArrayLike, rng_key: ArrayLike) -> (Array, Array, Array):
        indices = jax.random.permutation(rng_key, jnp.arange(x.shape[0]), independent=True)[:self.n_clusters]
        means = x[indices]
        # eye_mat = jnp.eye(self.n_clusters)

        def body(carry, _):
            means, x = carry

            diff_from_means = means[None, :] - x[:, None]
            dist_to_means = (diff_from_means ** 2).sum(-1)

            means_ids = dist_to_means.argmin(-1)
            # a = eye_mat[means_ids].T
            a = jax.nn.one_hot(means_ids, self.n_clusters).T

            means = (a @ x) / (a.sum(-1, keepdims=True) + 1e-10)#.clip(1e-10, jnp.inf)

            return (means, x), None

        (means, _), _ = jax.lax.scan(body, (means, x), None, length=self.n_iter)

        # After the final iteration, compute the final labels and inertia
        diff_from_means = means[None, :, :] - x[:, None, :]
        dist_to_means = (diff_from_means ** 2).sum(-1)
        means_ids = dist_to_means.argmin(-1)

        return means, means_ids, dist_to_means.min(-1).sum(-1)

    def fit_single(self, x: ArrayLike, rng_key=None):
        if rng_key is None:
            rng_key = make_jax_rng_key()

        results = self.fit(x, rng_key)

        self.cluster_centers_, self.labels_, self.inertia_ = results

    def fit_batched(self, x: ArrayLike, rng_key=None):
        if rng_key is None:
            rng_key = make_jax_rng_key()

        # keys = jax.random.split(key, num=X.shape[0])
        results = jax.vmap(self.fit, in_axes=(0, None))(x, rng_key)

        self.cluster_centers_, self.labels_, self.inertia_ = results


# key = jax.random.key(0)
# batch_size = 100
# n_points = 2500
# n_clusters = 128
# max_iter = 500
#
# X = jax.random.normal(key, (batch_size, n_points, 3))
#
# kmeans = JaxKMeans(n_clusters=n_clusters, max_iter=max_iter)
#
# for i in range(100):
#     print(i)
#     # X = jax.random.normal(key, (batch_size, n_points, 3))
#     key = jax.random.key(1)
#     kmeans.fit(X, key)
#
# print("Cluster Centers:\n", kmeans.cluster_centers_)
# print("Labels:\n", kmeans.labels_)
# print("Inertia:\n", kmeans.inertia_)
# print("Inertia:\n", kmeans.inertia_.mean())
