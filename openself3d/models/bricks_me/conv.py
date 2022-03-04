
import torch.nn as nn 
import MinkowskiEngine as ME 

from .registry import CONV_LAYERS_ME

CONV_LAYERS_ME.register_module('ME_Conv', module= ME.MinkowskiConvolution)
CONV_LAYERS_ME.register_module('ME_cwConv', module=ME.MinkowskiChannelwiseConvolution)

def build_conv_layer(cfg, *args, **kwargs):
    
    if cfg is None:
        cfg_ = dict(type='ME_Conv')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()
    
    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS_ME:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = CONV_LAYERS_ME.get(layer_type)
        if layer_type == 'ME_cwConv':
            if  'out_channels' in cfg_:
                cfg_.pop('out_channels')
            if  'out_channels' in  kwargs:
                kwargs.pop('out_channels')
            
    layer = conv_layer(*args,  **kwargs, **cfg_)
    
    return layer
