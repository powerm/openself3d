
import torch.nn as nn
import MinkowskiEngine as ME 

from .registry  import  UPSAMPLE_LAYERS_ME


UPSAMPLE_LAYERS_ME.register_module('ME_convT', module=ME.MinkowskiConvolutionTranspose)
UPSAMPLE_LAYERS_ME.register_module('ME_GconvT', module=ME.MinkowskiGenerativeConvolutionTranspose )



def  build_upsample_layer(cfg, *args, **kwargs):
    
    
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "type", but got{cfg}')
    cfg_ = cfg.copy()
    
    layer_type = cfg_.pop('type')
    if layer_type not in UPSAMPLE_LAYERS_ME:
        raise KeyError(f'Unrecongnized upsample type{layer_type}')
    else:
        upsample = UPSAMPLE_LAYERS_ME.get(layer_type)
    
    layer = upsample(*args, **kwargs, **cfg_)
    return layer
    