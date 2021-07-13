import flax.linen as nn
import jax.numpy as jnp
from chex import Array

from .utils import Sequential, gelu

__all__ = [
    "Affine",
    "CrossPatchSublayer",
    "CrossChannelSublayer",
    "ResMLPLayer",
    "ResMLP",
]

# ================ Helpers ===================
def full(key, shape, fill_value, dtype=None):
    return jnp.full(shape, fill_value, dtype)


def ones(key, shape, dtype=None):
    return jnp.ones(shape, dtype)


def zeros(key, shape, dtype=None):
    return jnp.zeros(shape, dtype)


# =============== Various Components ==============
class Affine(nn.Module):
    """
    A Flax linen Module to perform a Affine Transformation
    Attributes:
        dim: Needed to generate matrices of the appropriate shape
    """

    dim: int = 512

    def setup(self):
        self.alpha = self.param("alpha", ones, (1, 1, self.dim))
        self.beta = self.param("beta", zeros, (1, 1, self.dim))

    @nn.compact
    def __call__(self, x) -> Array:
        return x * self.alpha + self.beta


class CrossPatchSublayer(nn.Module):
    """
    A Flax linen Module consisting of two Affine element-wise transformations,
    Linear Layer and Skip Connection.

    Attributes:
        dim: dimensionality for the Affine Layer
        patch_size: dimensionality for the Linear Layer
        layerscale: float value for scaling the output
    """

    dim: int = 512
    patch_size: int = 16
    layerscale: float = 0.1

    def setup(self):
        self.aff1 = Affine(dim=self.dim)
        self.linear = nn.Dense(features=self.patch_size)
        self.aff2 = Affine(self.dim)
        self.layerscale = self.param(
            "layerscale_crosspatch", full, self.dim, self.layer_scale
        )

    @nn.compact
    def __call__(self, x) -> Array:
        # Output from Affine Layer 1
        transform = self.aff1(x)
        # Transpose the Affine Layer 1
        transposed_transform = jnp.transpose(transform, axes=(1, 2))
        # Feed into Linear Layer
        linear_transform = self.linear(transposed_transform)
        # Tranpose the output from Linear Layer
        transposed_linear = jnp.transpose(linear_transform, axes=(1, 2))
        # Feed into Affine Layer 2
        affine_output = self.aff2(transposed_linear)
        # Skip-Connection with LayerScale
        output = x + affine_output * self.layerscale

        return output


class CrossChannelSublayer(nn.Module):
    """
    A Flax linen Module consisting of two Affine element-wise transformations,
    MLP and Skip Connection.

    Attributes:
        dim: dimensionality for the Affine Layer and MLP fully-connected layers
        layerscale: float value for scaling the output
    """

    dim: int = 512
    layerscale: float = 0.1
    expansion_factor: int = 4

    def setup(self):
        self.aff1 = Affine(self.dim)
        self.mlp = Sequential(
            [
                nn.Dense(features=self.expansion_factor * self.dim),
                gelu(),
                nn.Dense(features=self.dim),
            ]
        )
        self.aff2 = Affine(self.dim)
        self.layerscale = self.param(
            "layerscale_crosschannel", full, self.dim, self.layer_scale
        )

    @nn.compact
    def __call__(self, x) -> Array:
        # Output from Affine Layer 1
        transform = self.aff1(x)
        # Feed into the MLP Block
        mlp_output = self.mlp(transform)
        # Output from Affine Layer 2
        affine_output = self.aff2(mlp_output)
        # Skip-Connection with LayerScale
        output = x + affine_output * self.layerscale

        return output


class ResMLPLayer(nn.Module):
    """
    A Flax linen Module consisting of the CrossPatchSublayer and CrossChannelSublayer

    Attributes:
        dim: dimensionality for the Affine and MLP layers
        depth: No of ResMLP Layers, needed for determining the layerscale value
        patch_size: dimensionality for the Linear Layer
    """

    dim: int = 512
    depth: int = 12
    patch_size: int = 16

    def setup(self):
        # Determine Value of LayerScale based on the depth
        if self.depth <= 18:
            layerscale = 0.1
        elif self.depth > 18 and self.depth <= 24:
            layerscale = 1e-5
        else:
            layerscale = 1e-6

        self.crosspatch = CrossPatchSublayer(
            dim=self.dim, patch_size=self.patch_size, layerscale=layerscale
        )
        self.crosschannel = CrossChannelSublayer(dim=self.dim, layerscale=layerscale)

    @nn.compact
    def __call__(self, x) -> Array:

        crosspatch_ouptput = self.crosspatch(x)
        crosschannel_output = self.crosschannel(crosspatch_ouptput)

        return crosschannel_output


# ================== Model ================
class ResMLP(nn.Module):
    """
    A Flax linen Module for creating the ResMLP architecture.

    Attributes:
        dim: dimensionality for the Affine and MLP layers
        depth: Number of ResMLP layers
        patch_size: dimensionality of the patches
        num_classes: No of classes
    """

    dim: int = 512
    depth: int = 12
    patch_size: int = 16
    num_classes: int = 10

    def setup(self):
        self.patch_projector = nn.Conv(
            features=self.dim, kernel_size=self.patch_size, strides=self.patch_size
        )

        self.blocks = Sequential(
            [
                ResMLPLayer(dim=self.dim, patch_size=self.patch_size, depth=self.depth)
                for _ in range(self.depth)
            ]
        )

        self.fc = nn.Dense(features=self.num_classes)

    @nn.compact
    def __call__(self, x) -> Array:
        x = self.patch_projector(x)
        x = self.blocks(x)
        output = jnp.mean(x, axis=1)
        return self.fc(output)
