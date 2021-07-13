import flax.linen as nn

from resmlp_flax.model import ResMLP


def test_instance():
    model = ResMLP(dim=512, depth=10, patch_size=16, num_classes=10)
    assert isinstance(model, nn.Module)
