from typing import Sequence

import jax.numpy as jnp
import numpy as np
from chex import Array
from flax import linen as nn
from jax import lax

__all__ = ["gelu", "Sequential"]


class gelu(nn.Module):
    approximate: bool = True

    @nn.compact
    def __call__(self, x) -> Array:
        if self.approximate:
            sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
            cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * (x ** 3))))
            return x * cdf
        else:
            return jnp.array(x * (lax.erf(x / np.sqrt(2)) + 1) / 2, dtype=x.dtype)


class Sequential(nn.Module):
    """
    Flax Module to act as a wrapper for creating Sequential Modules
    Attributes:
        layers: A Sequence of Flax Modules
    """

    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
