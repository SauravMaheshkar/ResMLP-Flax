from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from chex import Array
from einops import rearrange

from .utils import Reduce, Sequential, gelu

__all__ = ["Affine", "PreAffinePostLayerScale", "ResMLP"]

# ================ Helpers ====================
def full(key, shape, fill_value, dtype=None):
    return jnp.full(shape, fill_value, dtype)


def ones(key, shape, dtype=None):
    return jnp.ones(shape, dtype)


def zeros(key, shape, dtype=None):
    return jnp.zeros(shape, dtype)


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val


class Affine(nn.Module):
    dim: int

    def setup(self):
        self.g = self.param("g", ones, (1, 1, self.dim))
        self.b = self.param("b", zeros, (1, 1, self.dim))

    @nn.compact
    def __call__(self, x) -> Array:
        return x * self.g + self.b


class PreAffinePostLayerScale(nn.Module):
    dim: int
    depth: int
    fn: Any

    def setup(self):
        if self.depth <= 18:
            init_eps = 0.1
        elif self.depth > 18 and self.depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        self.scale = self.param("scale", full, (1, 1, self.dim), init_eps)
        self.affine = Affine(self.dim)

    @nn.compact
    def __call__(self, x):
        return self.fn(self.affine(x)) * self.scale + x


class ResMLP(nn.Module):

    dim: int
    depth: int
    image_size: Any
    patch_size: Any
    num_classes: int
    expansion_factor: int = 4

    def setup(self):
        image_height, image_width = pair(self.image_size)
        assert (image_height % self.patch_size) == 0 and (
            image_width % self.patch_size
        ) == 0, "image height and width must be divisible by patch size"
        self.num_patches = (image_height // self.patch_size) * (
            image_width // self.patch_size
        )
        self.wrapper = lambda i, fn: PreAffinePostLayerScale(self.dim, i + 1, fn)

        self.model = Sequential(
            [
                nn.Dense(features=self.dim),
                *[
                    Sequential(
                        [
                            self.wrapper(
                                i, nn.Conv(features=self.num_patches, kernel_size=1)
                            ),
                            self.wrapper(
                                i,
                                Sequential(
                                    [
                                        nn.Dense(
                                            features=self.dim * self.expansion_factor
                                        ),
                                        gelu(),
                                        nn.Dense(features=self.dim),
                                    ]
                                ),
                            ),
                        ]
                    )
                    for i in range(self.depth)
                ],
                Affine(dim=self.dim),
                Reduce("b n c -> b c", "mean"),
                nn.Dense(features=self.num_classes),
            ]
        )

    @nn.compact
    def __call__(self, x) -> Array:

        temp = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

        output = self.model(temp)

        return output
