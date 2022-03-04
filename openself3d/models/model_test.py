import torch 
import torch.nn as nn 
import mmcv 
import os 
#from builder import build_backbone
import torchvision.models as models
from mmcv.cnn import ResNet
from torch.autograd import Variable 
from  openself3d.datasets  import SpartanDataSource, prepare_config
from  openself3d.datasets import  ContrastiveDataset
from  dense_correspondence_network  import DenseCorrespondenceNetwork
from  fusenet_model import FuseNet

data_dir = '/media/cyn/e45903dd-cd53-44c9-8622-c21e80814317/whl/dataset/dense-net-entire/pdc/logs_proto'
configRoot = '/home/cyn/code/openself3d/config/dense_correspondence'
dataSetconfigRoot = os.path.join(configRoot, 'dataset')
config_file = '4_shoes_all.yaml'
dataconfig = prepare_config(dataSetconfigRoot, config_file,  data_dir)
train_config = mmcv.load(os.path.join(configRoot, 'training', 'training.yaml'))
dataset = ContrastiveDataset(dataconfig,train_config, mode='train')
net = DenseCorrespondenceNetwork.from_config(train_config['dense_correspondence_network'],
                                                      load_stored_params=False)
net1 = FuseNet(3)
