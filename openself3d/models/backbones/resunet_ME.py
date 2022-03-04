import torch 
import  torch.nn as nn 


import MinkowskiEngine as ME 
import MinkowskiEngine.MinkowskiFunctional as MEF 

from ..bricks import  (ConvModule,UpConvBlock, build_conv_layer,build_activation_layer, 
                       build_norm_layer,UPSAMPLE_LAYERS_ME, build_upsample_layer)

from  ..builder import BACKBONES

class BasicBlockBase(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=3,
                conv_cfg = dict(type='ME_Conv'),
                norm_cfg = dict(type='BN'),
                act_cfg = dict(type='MinkowskiReLU'),
                ):
        super(BasicBlockBase, self).__init__()
        
        self.conv1 = build_conv_layer(conv_cfg,            
                                      in_channels,
                                      out_channels,
                                      kernel_size=3,
                                      stride=stride,
                                      dilation=dilation,
                                      dimension=dimension )
        _, self.norm1 = build_norm_layer(norm_cfg, out_channels, dimension=dimension)
        self.conv2 = build_conv_layer(conv_cfg,            
                                      out_channels,
                                      out_channels,
                                      kernel_size=3,
                                      stride=1,
                                      dilation=dilation,
                                      bias=False,
                                      dimension=dimension )
        _, self.norm2 =  build_norm_layer(norm_cfg, out_channels, dimension=dimension)
        self.downsample = downsample
        
    def forward(self, x):
        
        residual = x
        
        out = self.conv1(x)
        out= self.norm1(x)
        out = MEF.relu(out)
        
        out = self.conv2(out)
        out= self.norm2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual 
        out = MEF.relu(out)
        
        return out

@BACKBONES.register_module()
class  ResUNet(ME.MinkowskiNetwork):
    
    def __init__(self,
                 in_channels=3,
                 base_channels =32,
                 out_channels=32,
                 channels=[32, 64, 128, 256],
                 tr_channels = [32, 64, 64, 128],
                 strides=(1,2,2,2),
                 enc_dilations=(1,1,1,1),
                 dec_dilations =(1,1,1,1),
                 num_stages = 4,
                 normalize_feature=None,
                 D=3,
                block_norm_cfg = dict(type='BN'),
                conv_cfg = dict(type='ME_Conv'),
                norm_cfg = dict(type='BN'),
                act_cfg = dict(type='MinkowskiReLU'),
                 ):
        ME.MinkowskiNetwork.__init__(self, D)
        self.channels = channels
        self.tr_channels = tr_channels
        self.normalize_feature = normalize_feature
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        tr_channels.append(0)
        
        for i in range(num_stages):
            convblock = ConvModule(
                        in_channels,
                        channels[i],
                        3,
                        stride=strides[i],
                        dilation=enc_dilations[i],
                        conv_cfg = conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg =None)
            in_channels = channels[i]
            block = BasicBlockBase(channels[i],channels[i],norm_cfg =block_norm_cfg,dimension=3)
            self.encoder.append(nn.Sequential(convblock, block))
        
        self.conv4_tr = build_upsample_layer(
                dict(type='DeconvModule'),
                channels[3]+tr_channels[3+1],
                tr_channels[3],
                dimension=3,
                kernel_size = 3,
                dilation=dec_dilations[3],
                bias = 'auto',
                stride = strides[3],
                inplace= True,
                with_cp=False,
                up_cfg= dict(type='ME_convT'),
                norm_cfg =dict(type='BN'),
                act_cfg = None,
                order=('convT', 'norm', 'act'))
        self.block4_tr = BasicBlockBase(tr_channels[3], tr_channels[3],norm_cfg =block_norm_cfg,dimension=3)
        self.conv3_tr = build_upsample_layer(
                dict(type='DeconvModule'),
                channels[2]+tr_channels[3],
                tr_channels[2],
                dimension=3,
                kernel_size = 3,
                dilation=dec_dilations[2],
                bias = 'auto',
                stride = strides[2],
                inplace= True,
                with_cp=False,
                up_cfg= dict(type='ME_convT'),
                norm_cfg =dict(type='BN'),
                act_cfg = None,
                order=('convT', 'norm', 'act'))
        self.block3_tr = BasicBlockBase(tr_channels[2], tr_channels[2],norm_cfg =block_norm_cfg,dimension=3)

        self.conv2_tr = build_upsample_layer(
                dict(type='DeconvModule'),
                channels[1]+tr_channels[2],
                tr_channels[1],
                dimension=3,
                kernel_size = 3,
                dilation=dec_dilations[1],
                bias = 'auto',
                stride = strides[1],
                inplace= True,
                with_cp=False,
                up_cfg= dict(type='ME_convT'),
                norm_cfg =dict(type='BN'),
                act_cfg = None,
                order=('convT', 'norm', 'act'))
        self.block2_tr = BasicBlockBase(tr_channels[1],tr_channels[1],norm_cfg =block_norm_cfg,dimension=3)
        self.conv1_tr = ME.MinkowskiConvolution(
            in_channels=channels[0] + tr_channels[1],
            out_channels=tr_channels[0],
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        
        self.final = ME.MinkowskiConvolution(
                in_channels=tr_channels[0],
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=True,
                dimension=D)
        
    def  forward(self, x):
        
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
            x = MEF.relu(x)
            
        dec_outs = [x]
        
        out = self.conv4_tr(x)
        out = self.block4_tr(out)
        out_s4_tr  = MEF.relu(out)
        
        out = ME.cat(out_s4_tr,  enc_outs[2])
        
        out = self.conv3_tr(out)
        out = self.block3_tr(out)
        out_s2_tr  = MEF.relu(out)
        
        out = ME.cat(out_s2_tr,  enc_outs[1])
        
        out = self.conv2_tr(out)
        out = self.block2_tr(out)
        out_s1_tr  = MEF.relu(out)
        
        out = ME.cat(out_s1_tr,  enc_outs[0])
        
        out=self.conv1_tr(out)
        out = MEF.relu(out)
        out= self.final(out)
        
        if self.normalize_feature:
            return ME.SparseTensor(
                out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
                coordinate_map_key=out.coordinate_map_key,
                coordinate_manager=out.coordinate_manager)
        else:
            return out
        
        

if __name__ == 'main':

    model = ResUNet(3, 32, normalize_feature=True, D =3 )
        
        
        
        
            
        
        