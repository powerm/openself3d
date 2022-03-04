from torch import nn 
from  ..utils import  Registry, build_from_cfg


MODELS = Registry('model')
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
FUSION_LAYERS = Registry('fusion')
LOSSES = Registry('loss')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, it is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Default: None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_model(cfg):
    return build(cfg, MODELS)

def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)

def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)

def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)

def  build_fusion(cfg):
    """Build fusion layer

    Args:
        cfg ([type]): [description]
    """
    return build(cfg, FUSION_LAYERS)