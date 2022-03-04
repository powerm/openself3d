import inspect

import  MinkowskiEngine as ME 
from .registry import NORM_LAYERS_ME
from mmcv.utils import is_tuple_of

NORM_LAYERS_ME.register_module('BN', module=ME.MinkowskiBatchNorm)
NORM_LAYERS_ME.register_module('IN', module = ME.MinkowskiInstanceNorm)
NORM_LAYERS_ME.register_module('SyncBN', module=ME.MinkowskiSyncBatchNorm)



def infer_abbr(class_type):
    
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    class_name = class_type.__name__.lower()
    if 'instance' in class_name:
        return 'in'
    elif 'batch' in class_name:
        return 'bn'
        
    

def build_norm_layer(cfg, num_features, postfix='', **kwargs):
    
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS_ME:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    
    norm_layer = NORM_LAYERS_ME.get(layer_type)
    abbr = infer_abbr(norm_layer)
    
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    
    if layer_type =='IN':
        layer = norm_layer(num_features)
    else:
        if 'momentum' not in cfg_:
            cfg_.setdefault('momentum', 0.05)
        if layer_type == 'SyncBN':
            pass
        # assert 'num_groups' in cfg_
        layer = norm_layer(num_features, **cfg_)
    
    
        
    return name, layer

def is_norm(layer, exclude=None):
    
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not is_tuple_of(exclude, type):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False
   
    # modifing
    all_norm_bases = (ME.MinkowskiBatchNorm,ME.MinkowskiInstanceNorm, ME.MinkowskiSyncBatchNorm)
    return isinstance(layer, all_norm_bases)