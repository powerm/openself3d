
import torch.nn as nn
import  MinkowskiEngine as ME

from openself3d.models.bricks import build_upsample_layer,ConvModule, UPSAMPLE_LAYERS_ME
from . activation import build_activation_layer
from .norm import build_norm_layer


@UPSAMPLE_LAYERS_ME.register_module()
class  DeconvModule(nn.Module): 
  
    def __init__(self,
               in_channels,
               out_channels,
               dimension=3,
               kernel_size = 3,
               dilation =1,
               bias = 'auto',
               stride = 1,
               inplace= True,
               with_cp=False,
               up_cfg= dict(type='ME_convT'),
               norm_cfg =dict(type='BN'),
               act_cfg = dict(type='MinkowskiReLU'),
               order=('convT', 'norm', 'act')
               ):
        super(DeconvModule, self).__init__()
        assert up_cfg  is None or isinstance(up_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg  is None or isinstance(act_cfg, dict)
    
        self.up_cfg = up_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) ==3
        assert set(order) == set(['convT', 'norm', 'act'])
    
        self.with_norm = norm_cfg  is not None
        self.with_activation = act_cfg is not None
        
            # if the conv layer is before a norm layer, bias is unnecessary
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias
        
        self.with_cp = with_cp
        self.deconv = build_upsample_layer(up_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride =stride,
            dilation=dilation,
            bias=bias,
            dimension=dimension
        )
    
        self.out_channels = self.deconv.out_channels
        self.in_channels = self.deconv.in_channels
        self.kernel_size = self.deconv.kernel_generator.kernel_size
        self.stride = self.deconv.kernel_generator.kernel_stride
        self.dilation = self.deconv.kernel_generator.kernel_dilation
        self.dimension = self.deconv.dimension
        self.transposed = self.deconv.is_transpose
    
        #build normalization layers
        if self.with_norm:
                # norm layer is after conv layer
            if order.index('norm')> order.index('convT'):
                    norm_channels = out_channels
            else:
                    norm_channels = in_channels
            self.norm_name,  norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name,  norm)
        else:
            self.norm_name = None    

        
        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            if act_cfg_['type']  not in [
                        'MinkowskiTanh', 'MinkowskiPReLU', 'MinkowskiSigmoid',   
                        'MinkowskiHardsigmoid', 'MinkowskiHardswish' 
                ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)    
        
    @property
    def norm(self):
            if self.norm_name:
                return getattr(self, self.norm_name)
            else:
                return None
    
    def forward(self, x, activate=True, norm=True):
        
        for layer in self.order:
            if layer == 'convT':
                x = self.deconv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x



class UpConvBlock(ME.MinkowskiModuleBase):
    """Upsample convolution block in decoder for Unet
    
    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """
    
    
    
    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 kernel_size=3,
                conv_block=None,
                 stride =1,
                 dilation=1,
                 dimension=3,
                 with_cp =False,
                 conv_cfg = None,
                 norm_cfg =dict(type='BN'),
                 act_cfg=dict(type='MinkowskiReLU'),
                 upsample_cfg=dict(type='DeconvModule'),
                 dcn=None,
                 plugins = None):
        super(UpConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
    
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(
                cfg = upsample_cfg,
                in_channels = in_channels+skip_channels,
                out_channels=out_channels,
                kernel_size = kernel_size,
                stride =stride,
                dilation=dilation,
                dimension=dimension,
                with_cp = with_cp,
                norm_cfg = norm_cfg,
                act_cfg=act_cfg,
            )
        else:
            self.upsample = ConvModule(
                in_channels,
                out_channels,
                stride =stride,
                dilation=dilation,
                dimension=dimension,
                conv_cfg = conv_cfg,
                norm_cfg= norm_cfg,
                act_cfg=act_cfg
            )
            
        # self.conv_block = conv_block(
        #     in_channels = 2* skip_channels,
        #     out_channels = out_channels,
        #     num_convs = num_convs,
        #     stride = stride,
        #     dilation=dilation,
        #     dimension=dimension,
        #     with_cp = with_cp,
        #     conv_cfg = conv_cfg,
        #     norm_cfg = norm_cfg,
        #     act_cfg = act_cfg,
        #     dcn =None,
        #     plugins=None
        # )
    
    def forward(self, skip, x, cat=True):
        out = self.upsample(x)
        #out = self.conv_block(x)
        if cat:
            out = ME.cat(out,  skip)
        return out
    