import functools
from typing import Sequence

from einops.einops import EinopsError, TransformRecipe, _prepare_transformation_recipe
from flax import linen as nn

__all__ = ["Sequential", "Rearrange", "Reduce"]


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


class RearrangeMixin:
    """
    From einops/einops/layers/__init__.py
    """

    def __init__(self, pattern, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths
        self._recipe = self.recipe()  # checking parameters

    def __repr__(self):
        params = repr(self.pattern)
        for axis, length in self.axes_lengths.items():
            params += ", {}={}".format(axis, length)
        return "{}({})".format(self.__class__.__name__, params)

    @functools.lru_cache(maxsize=1024)
    def recipe(self) -> TransformRecipe:
        try:
            hashable_lengths = tuple(sorted(self.axes_lengths.items()))
            return _prepare_transformation_recipe(
                self.pattern, operation="rearrange", axes_lengths=hashable_lengths
            )
        except EinopsError as e:
            raise EinopsError(" Error while preparing {!r}\n {}".format(self, e))

    def _apply_recipe(self, x):
        return self._recipe.apply(x)


class ReduceMixin:
    """
    From einops/einops/layers/__init__.py
    """

    def __init__(self, pattern, reduction, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction
        self.axes_lengths = axes_lengths
        self._recipe = self.recipe()  # checking parameters

    def __repr__(self):
        params = "{!r}, {!r}".format(self.pattern, self.reduction)
        for axis, length in self.axes_lengths.items():
            params += ", {}={}".format(axis, length)
        return "{}({})".format(self.__class__.__name__, params)

    @functools.lru_cache(maxsize=1024)
    def recipe(self) -> TransformRecipe:
        try:
            hashable_lengths = tuple(sorted(self.axes_lengths.items()))
            return _prepare_transformation_recipe(
                self.pattern, operation=self.reduction, axes_lengths=hashable_lengths
            )
        except EinopsError as e:
            raise EinopsError(" Error while preparing {!r}\n {}".format(self, e))

    def _apply_recipe(self, x):
        return self._recipe.apply(x)


class Rearrange(nn.Module):
    """
    Flax Module to act as a Rearrange layer (from einops)
    """

    pattern: str

    def setup(self):
        self.rearranger = RearrangeMixin(self.pattern)

    @nn.compact
    def __call__(self, input):
        return self.rearranger._apply_recipe(input)


class Reduce(nn.Module):
    pattern: str
    reduction: str

    def setup(self):
        self.reducer = ReduceMixin(self.pattern, self.reduction)

    @nn.compact
    def __call__(self, input):
        return self.reducer._apply_recipe(input)
