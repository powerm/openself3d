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
from  openself3d.models  import DenseCorrespondenceNetwork
from  openself3d.models import FuseNet
from openself3d.apis   import  DCNTraining

data_dir = '/media/cyn/e45903dd-cd53-44c9-8622-c21e80814317/whl/dataset/dense-net-entire/pdc/logs_proto'
configRoot = '/home/cyn/code/openself3d/config/dense_correspondence'
dataSetconfigRoot = os.path.join(configRoot, 'dataset')
config_file = 'baymax_only.yaml'
dataconfig = prepare_config(dataSetconfigRoot, config_file,  data_dir)
train_config = mmcv.load(os.path.join(configRoot, 'training', 'training.yaml'))
dataset = ContrastiveDataset(dataconfig,train_config, mode='train')
net = DenseCorrespondenceNetwork.from_config(train_config['dense_correspondence_network'],
                                                      load_stored_params=False)
net1 = FuseNet(3, use_class=False)

train = DCNTraining(net1,dataset ,train_config)

train.run(two_input=True)

