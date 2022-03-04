import warnings
#from mmcv.cnn.bricks  import UPSAMPLE_LAYERS_ME
import torch 
import torch.nn as nn

import torch.utils.checkpoint as cp

from ..bricks import  (ConvModule,UpConvBlock, build_activation_layer, 
                       build_norm_layer,UPSAMPLE_LAYERS_ME,DeconvModule)

import MinkowskiEngine as ME 
import MinkowskiEngine.MinkowskiFunctional as MEF 

from  ..builder import BACKBONES

import numpy as np 

class BasicBlockBase(nn.Module):
  
  def __init__(self,
               in_channels,
               out_channels,
               num_convs = 2,
               stride = 1,
               dilation =1,
               dimension = 3,
               bn_momentum=0.1,
               with_cp = False,
               conv_cfg = None,
               norm_cfg = dict(type='BN'),
               act_cfg = dict(type='MinkowskiReLU'),
               dcn =None,
               plugins = None
               ):
    super(BasicBlockBase, self).__init__()
    assert dcn is None, 'Not implemented yet.'
    assert plugins is None, 'Not implemented yet.'
    norm_cfg.setdefault('momentum', bn_momentum)
    
    self.with_cp = with_cp 
    convs = []
    for i in range(num_convs):
      convs.append(
        ConvModule(
          in_channels = in_channels if i==0 else out_channels,
          out_channels=out_channels,
          kernel_size =3,
          stride= stride if i==0 else 1,
          dilation = 1 if i==0 else  dilation,
          conv_cfg = conv_cfg,
          norm_cfg = norm_cfg,
          act_cfg=act_cfg))
    
    self.convs = nn.Sequential(*convs)
    
  def forward(self, x):
    
    if self.with_cp and x.requires_grad:
      out = cp.checkpoint(self.convs, x)
    else:
      out = self.convs(x)
    return out
  

@BACKBONES.register_module()
class UNet_ME(nn.Module):
  
  def __init__(self,
              in_channels = 3,
              base_channels = 32,
              num_stages = 5,
              dimension=3,
              strides = (1, 2, 2, 2, 2),
              enc_num_convs =(2,2,2,2,2),
              dec_num_convs =(2,2,2,2,2),
              downsamples = (True,True,True,True,True),
              enc_dilations =(1,1,1,1,1),
              dec_dilations =(1,1,1,1,1),
              with_cp = False,
              conv_cfg = None,
              norm_cfg = dict(type='BN'),
              act_cfg = dict(type='MinkowskiReLU'),
              upsample_cfg = dict(type='DeconvModule'),
              norm_eval = False,
              dcn = None,
              plugins = None,
              pretrained = None,
              init_cfg = None):

    super(UNet_ME, self).__init__()
    
    self.pretrained = pretrained
    assert not (init_cfg and pretrained), \
        'init_cfg and pretrained cannot be setting at the same time'
    if isinstance(pretrained, str):
      warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                    'please use "init_cfg" instead')
      self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
    elif pretrained is None:
      if init_cfg  is None:
        self.init_cfg = [
          dict(type='Kaiming', layer='Conv2d'),
          dict(
            type='Constant',
            val=1,
            layer=['_BatchNorm', 'GroupNorm'])
         ]
      else:
        raise TypeError('pretrained must be a str or None')
      
    assert dcn is None, 'Not implemented yet.'
    assert plugins  is None, 'Not implemented yet.'
    assert len(strides) == num_stages, \
      'The length of strides should be equal to num_stages, '\
        f'while the strides is {strides}, the length of '\
        f'strides is {len(strides)}, and the num_stages is '\
        f'{num_stages}.'
    assert len(enc_num_convs) == num_stages, \
      'The length of enc_num_convs should be equal to num_stages, '\
      f'while the enc_num_convs is {enc_num_convs}, the length of '\
      f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is '\
      f'{num_stages}.'
    assert len(dec_num_convs) == (num_stages), \
      'The length of dec_num_convs should be equal to (num_stages-1), '\
      f'while the dec_num_convs is {dec_num_convs}, the length of'\
      f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is '\
      f'{num_stages}.'
    assert len(downsamples) == (num_stages), \
      'The length of downsamples should be equal to (num_stages-1),'\
      f'while the downsamples is {downsamples}, the length of '\
      f'downsamples is {len(downsamples)}, and the num_stages is '\
      f'{num_stages}.'
    assert len(enc_dilations) == num_stages,\
      'The length of enc_dilations should be equal to num_stages,' \
      f'while the enc_dilation is {enc_dilations}, the length of '\
      f'enc_dilation is {len(enc_dilations)}, and the num_stages is '\
      f'{num_stages}.'
    assert len(dec_dilations) == (num_stages), \
      'The length of dec_dilations should be equal to (num_stages-1),'\
      f'while the dec_dilations is {dec_dilations}, the length of'\
      f'dec_dilations is {len(dec_dilations)}, and the num_stages is '\
      f'{num_stages}.'
    
    self.num_stages = num_stages
    self.strides = strides
    self.downsamples = downsamples
    self.norm_eval = norm_eval
    self.base_channels = base_channels
    
    self.encoder = nn.ModuleList()
    self.decoder = nn.ModuleList()
    
    self.act = build_activation_layer(act_cfg)
    TR_CHANNELS = []
    CHANNELS = []
    for  i in range(num_stages):
      CHANNELS.append(base_channels*2**i)
      TR_CHANNELS.append(base_channels*2**(i//2))
    TR_CHANNELS.append(0)
    
    for i in range(num_stages):
      upsample = (strides[i] != 1 or downsamples[i])
      self.decoder.append(
        DeconvModule( TR_CHANNELS[i+1]+CHANNELS[i],TR_CHANNELS[i],
                     dimension=dimension,
                     dilation=dec_dilations[i],
                     stride=strides[i],
                     with_cp=with_cp,
                     norm_cfg = norm_cfg,
                     act_cfg=act_cfg))
        
          # UpConvBlock(
          #   in_channels= TR_CHANNELS[i+1],
          #   skip_channels = CHANNELS[i],
          #   out_channels= TR_CHANNELS[i],
          #   kernel_size=3,
          #   stride=strides[i],
          #   dilation= dec_dilations[i],
          #   dimension=3,
          #   conv_block=BasicBlockBase,
          #   with_cp=with_cp,
          #   conv_cfg=conv_cfg,
          #   norm_cfg=norm_cfg,
          #   act_cfg=act_cfg,
          #   upsample_cfg=upsample_cfg if upsample else None,
          #   dcn=None,
          #   plugins = None))
        
      #enc_conv_block.append(
      self.encoder.append(
        ConvModule(
            in_channels,
            base_channels*2**i,
            3,
            stride=strides[i],
            dilation=enc_dilations[i],
            conv_cfg = conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg =None))
      #self.encoder.append(*enc_conv_block)
      in_channels = base_channels*2**i

  def forward(self,   x):
    #self._check_input_divisible(x)
    enc_outs = []
    for enc in self.encoder:
      x = enc(x)
      enc_outs.append(x)
      x = self.act(x)
    dec_outs = [x]
    for i in reversed(range(len(self.decoder))):
      if i==0:
        break
      x = self.decoder[i](x)
      dec_outs.append(x)
      x=ME.cat(x,  enc_outs[i-1])
    
    out= dec_outs[0][0](x)
    
    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coordinate_map_key=out.coordinate_map_key,
          coordinate_manager=out.coordinate_manager)
    else:
      return out
    
    return dec_outs


# if __name__ == "__main__":
  
#   unet = UNet_ME(
#         in_channels=3,
#         base_channels=64,
#         num_stages=5,
#         strides=(1, 1, 1, 1, 1),
#         enc_num_convs=(2, 2, 2, 2, 2),
#         dec_num_convs=(2, 2, 2, 2),
#         downsamples=(True, True, True, True),
#         enc_dilations=(1, 1, 1, 1, 1),
#         dec_dilations=(1, 1, 1, 1))
    
    
#   origin_pc1 = 100*np.random.uniform(0,1, (10,3))
#   feat1 = np.ones((10,32), dtype=np.float32)
#   origin_pc2 = 100*np.random.uniform(0,1,(6,3))
#   feat2 = np.ones((6,32), dtype = np.float32)
#   coords, feats = ME.utils.sparse_collate([origin_pc1, origin_pc2], [feat1, feat2])
#   x = ME.SparseTensor(feats,  coordinates = coords)
    
#   print(unet)

#   x_outs = unet(x)