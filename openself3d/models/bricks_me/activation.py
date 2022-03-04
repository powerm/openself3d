import MinkowskiEngine as ME 
import MinkowskiEngine.MinkowskiFunctional as MEF

from mmcv.utils import  build_from_cfg

from .registry import ACTIVATION_LAYERS_ME

for module in [
    ME.MinkowskiReLU,  ME.MinkowskiLeakyReLU,  ME.MinkowskiPReLU,
    ME.MinkowskiRReLU, ME.MinkowskiReLU6,  ME.MinkowskiELU, 
    ME.MinkowskiSigmoid, ME.MinkowskiTanh,  
    ME.MinkowskiHardsigmoid,ME.MinkowskiHardswish,
    ME.MinkowskiHardtanh
    ]:
    ACTIVATION_LAYERS_ME.register_module(module=module)


def build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return build_from_cfg(cfg, ACTIVATION_LAYERS_ME)