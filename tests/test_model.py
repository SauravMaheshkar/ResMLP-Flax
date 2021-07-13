import flax.linen as nn
from jax import random

from resmlp_flax.model import Affine, ResMLP


def test_instance():
    aff = Affine(dim=64)
    model = ResMLP(image_size=256, patch_size=16, dim=512, depth=12, num_classes=10)
    assert isinstance(aff, nn.Module)
    assert isinstance(model, nn.Module)


def test_model():

    x = random.normal(key=random.PRNGKey(0), shape=(1, 3, 256, 256))
    ResMLP(image_size=256, patch_size=16, dim=256, depth=12, num_classes=1000).init(
        random.PRNGKey(1), x
    )
