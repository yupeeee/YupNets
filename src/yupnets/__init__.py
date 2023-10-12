from .nets import *
from .utils import *
from .xray import *

__all__ = [
    # nets
    "load_net",

    # utils
    "Activation",
    "BatchManager",
    "Normalization",

    # xray
    "get_layer_ids",
    "get_layer",
    "FeatureExtractor",
]


def load_net(
        net: str,
        **kwargs,
):
    if net in resnet.nets or net == "resnet":
        model = resnet.resnet_loader(net=net, **kwargs)

    elif net in vit.nets or net == "vit":
        model = vit.vit_loader(net=net, **kwargs)

    else:
        raise ValueError(f"Unsupported net: {net}")

    return model
