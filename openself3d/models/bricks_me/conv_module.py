import  warnings 

from torch import nn
from . activation import build_activation_layer
from .conv import build_conv_layer
from .norm import build_norm_layer
from ..utils import  kaiming_init, constant_init

from .registry import  PLUGIN_LAYERS_ME


@PLUGIN_LAYERS_ME.register_module()
class ConvModule(nn.Module):
    
    
    _abbr_ = 'conv_block'
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias = 'auto',
                 kernel_generator=None,
                 expand_coordinates=False,
                 dimension=3,
                 inplace = True,
                 conv_cfg = None,
                 norm_cfg = None,
                 act_cfg = dict(type='MinkowskiReLU'),
                 order =('conv','norm', 'act')
                 ):
        super(ConvModule,self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) ==3
        assert set(order) == set(['conv', 'norm', 'act'])
        
        self.with_norm = norm_cfg  is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        self.conv = build_conv_layer(
            conv_cfg, 
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            kernel_generator=kernel_generator,
            dimension=dimension   
            )
        
        self.out_channels = self.conv.out_channels
        self.in_channels = self.conv.in_channels
        self.kernel_size = self.conv.kernel_generator.kernel_size
        self.stride = self.conv.kernel_generator.kernel_stride
        self.dilation = self.conv.kernel_generator.kernel_dilation
        self.dimension = self.conv.dimension
        self.transposed = self.conv.is_transpose
        
        #build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm')> order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name,  norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
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
        
        # Use msra init by default
        self.init_weights()
        
    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None
        
    def init_weights(self):
            
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'MinkowskiLeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a=0
                kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias =0)
            
    def  forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif  layer =='act' and activate  and self.with_activation:
                x = self.activate(x)
        return x
                    
            
            

        
        
        




