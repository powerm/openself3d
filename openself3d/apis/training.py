
import numpy as np 
import os 
import fnmatch 
import gc 
import logging 
import time 

import shutil 

import subprocess 
import copy 

#  torch
import torch 
from torchvision import transforms 
from torch.autograd import Variable 
import torch.nn as nn 
import torch.optim as optim 

import mmcv

#
import tensorboard_logger


from openself3d.utils import  get_root_logger,  get_current_time_unique_name, get_unique_string, getPaddedString
from openself3d.losses  import PixelwiseContrastiveLoss, get_loss, is_zero_loss 
from openself3d.datasets import SpartanDatasetDataType


def build_dataloader(dataset , cfg):
    return  torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, 
                                              num_workers=cfg['num_workers'], drop_last=True)

def build_optimizer(model,  optimizer_cfg):
    
    if hasattr(model, 'module'):
        model = model.module
    learning_rate = float(optimizer_cfg['training']['learning_rate'])
    weight_decay = float(optimizer_cfg['training']['weight_decay'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def setup_logger(cfg):
    """
        Sets up the directory where logs will be stored and config
        files written
        :return: full path of logging dir
        :rtype: str
    """
    if 'logging_dir_name' in cfg['log']:
        dir_name = cfg['log']['logging_dir_name']
    else: 
        dir_name = get_current_time_unique_name() + "_" + str(cfg['dense_correspondence_network']['descriptor_dimension']) + "d"
        
    pdc_path = os.path.dirname(cfg['dataset']['root'])
    logging_dir = os.path.join(pdc_path, cfg['log']['logging_dir'],  dir_name)
    print('logging_dir:', logging_dir)
    
    if os.path.isdir(logging_dir):
        shutil.rmtree(logging_dir)
    if not os.path.isdir(logging_dir):
            os.makedirs(logging_dir)
    # make the tensorboard log directory
    tensorboard_log_dir = os.path.join(logging_dir, "tensorboard")
    if not os.path.isdir(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    return logging_dir, tensorboard_log_dir

def setup_tensorboard(tensorboard_log_dir):
    # start tensorboard
    logging.info("setting up tensorboard_logger")
    cmd = "tensorboard --logdir=%s" %(tensorboard_log_dir)
    tb_logger = tensorboard_logger.Logger(tensorboard_log_dir)
    return tb_logger
    

class  DCNTraining(object):
    
    def __init__(self,
                 model,
                 dataset,
                 cfg,
                 distributed=False,
                 validate = False,
                 optimaizer=None,
                 logger=None,
                 meta=None,
                 max_epochs=80,
                 ):
        super(DCNTraining, self).__init__()
        self.cfg = cfg
        self.max_epochs = max_epochs
        self.logger = get_root_logger(cfg['log']['log_level'])
        self.logging_dir, self.tensorboard_log_dir = setup_logger(cfg)
        self._tensorboard_logger = setup_tensorboard(self.tensorboard_log_dir)
        # prepare data loaders
        self.dataset = dataset
        self.data_loader =  build_dataloader(self.dataset, cfg['dataloader'])
        # build optimizer
        self.optimizer = build_optimizer(model, cfg)
        self.model = model
        self.descriptor_dimension = cfg['dense_correspondence_network']['descriptor_dimension']
        self.img_width = cfg['dense_correspondence_network']['image_width']
        self.img_height = cfg['dense_correspondence_network']['image_height']
        self.img_shape = [self.img_height, self.img_width]
        self.loss_func = PixelwiseContrastiveLoss(image_shape=self.img_shape, config=self.cfg['loss_function'])
        self.loss_func.debug = True
        self.num_iterations = self.cfg['training']['num_iterations']
        self.logging_rate = self.cfg['log']['logging_rate']
        self.save_rate = self.cfg['log']['save_rate']
        self.compute_test_loss_rate = self.cfg['training']['compute_test_loss_rate']
        # logging
        self.logging_dict = dict()
        self.logging_dict['train'] = {"iteration": [], "loss": [], "match_loss": [],
                                           "masked_non_match_loss": [], 
                                           "background_non_match_loss": [],
                                           "blind_non_match_loss": [],
                                           "learning_rate": [],
                                           "different_object_non_match_loss": []}

        self.logging_dict['test'] = {"iteration": [], "loss": [], "match_loss": [],
                                           "non_match_loss": []}
        
    
    def save_configs(self):
        # save the config file to the logging directory
        training_params_file = os.path.join(self.logging_dir, 'training.yaml')
        dataset_params_file = os.path.join(self.logging_dir, 'dataset.yaml')
        mmcv.dump(self.cfg, training_params_file)
        mmcv.dump(self.dataset.dataSource._config, dataset_params_file)
        # make unique identifier
        identifier_file = os.path.join(self.logging_dir, 'identifier.yaml')
        identifier_dict = dict()
        identifier_dict['id'] = get_unique_string()
        mmcv.dump(identifier_dict, identifier_file)
    
    def _get_current_loss(self, logging_dict):
        """
        Gets the current loss for both test and train
        :return:
        :rtype: dict
        """
        d = dict()
        d['train'] = dict()
        d['test'] = dict()

        #for key, val in d.iteritems():
        for key, val in iter(d.items()):
            for field in logging_dict[key].keys():
                vec = logging_dict[key][field]

                if len(vec) > 0:
                    val[field] = vec[-1]
                else:
                    val[field] = -1 # placeholder
        return d
    
    def save_network(self, model, optimizer, iteration, logging_dict=None):
        #  Saves network parameters to logging directory
        network_param_file = os.path.join(self.logging_dir,  getPaddedString(iteration, width=6) + ".pth")
        optimizer_param_file = network_param_file + ".opt"
        torch.save(model.state_dict(), network_param_file)
        torch.save(optimizer.state_dict(), optimizer_param_file)
        
        # also save loss history stuff
        if logging_dict is not None:
            log_history_file = os.path.join(self.logging_dir,  getPaddedString(iteration, width=6) + "_log_history.yaml")
            mmcv.dump(logging_dict, log_history_file)
            current_loss_file = os.path.join(self.logging_dir, 'loss.yaml')
            current_loss_data = self._get_current_loss(logging_dict)
            mmcv.dump(current_loss_data, current_loss_file)
    
    def resume(self, model_folder, iteration = None):
        """
        Loads network and optimizer parameters from a previous training run.
        Note: It is up to the user to ensure that the model parameters match.
        e.g. width, height, descriptor dimension etc.
        :param model_folder: location of the folder containing the param files 001000.pth. Can be absolute or relative path. If relative then it is relative to pdc/trained_models/
        :type model_folder:
        :param iteration: which index to use, e.g. 3500, if None it loads the latest one
        :type iteration:
        :return: iteration
        :rtype:
        """
        if not os.path.isdir(model_folder):
            pdc_path = os.path.dirname(self.cfg['dataset']['root'])
            model_folder = os.path.join(pdc_path, "trained_models", model_folder)
        # find idx.pth and idx.pth.opt files
        if iteration is None:
            files = os.listdir(model_folder)
            model_param_file = sorted(fnmatch.filter(files, '*.pth'))[-1]
            iteration = int(model_param_file.split(".")[0])
            optim_param_file = sorted(fnmatch.filter(files, '*.pth.opt'))[-1]
        else:
            prefix = getPaddedString(iteration, width=6)
            model_param_file = prefix + ".pth"
            optim_param_file = prefix + ".pth.opt"
        print ("model_param_file", model_param_file)
        model_param_file = os.path.join(model_folder, model_param_file)
        optim_param_file = os.path.join(model_folder, optim_param_file)
        
        self.model.load_state_dict(torch.load(model_param_file))
        self.model.cuda()
        self.model.train()
        self.optimizer.load_state_dict(torch.load(optim_param_file))
        
        return iteration
        
        
        
    
    def adjust_learning_rate(self, optimizer, iteration):
        """
        Adjusts the learning rate according to the schedule
        :param optimizer:
        :type optimizer:
        :param iteration:
        :type iteration:
        :return:
        :rtype:
        """

        steps_between_learning_rate_decay = self.cfg['training']['steps_between_learning_rate_decay']
        if iteration % steps_between_learning_rate_decay == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.cfg["training"]["learning_rate_decay"]
       
    def process_net_output(self, img_pre, N):
        W = self.img_width
        H = self.img_height
        image_pred = img_pre.view(N, self.descriptor_dimension, W * H)
        image_pred = image_pred.permute(0, 2, 1)
        return image_pred
    

    def get_learning_rate(self, optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr
    
    def update_log(self, loss_current_iteration,learning_rate, loss, match_loss, \
                   masked_non_match_loss, background_non_match_loss, blind_non_match_loss, data_type):
        # Updates the tensorboard plots with current loss function information
        self.logging_dict['train']['learning_rate'].append(learning_rate)
        self._tensorboard_logger.log_value("learning rate", learning_rate, loss_current_iteration)
        
                    # Don't update any plots if the entry corresponding to that term
                    # is a zero loss
        if not  is_zero_loss(match_loss):
            self.logging_dict['train']['match_loss'].append(match_loss.item())
            self._tensorboard_logger.log_value("train match loss", match_loss.item(), loss_current_iteration)
        
        if not is_zero_loss(masked_non_match_loss):
            self.logging_dict['train']['masked_non_match_loss'].append(masked_non_match_loss.item())
            self._tensorboard_logger.log_value("train masked non match loss", masked_non_match_loss.item(), loss_current_iteration)

        if not is_zero_loss(background_non_match_loss):
            self.logging_dict['train']['background_non_match_loss'].append(background_non_match_loss.item())
            self._tensorboard_logger.log_value("train background non match loss", background_non_match_loss.item(), loss_current_iteration)

        if not is_zero_loss(blind_non_match_loss):
            if data_type == SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE:
                self._tensorboard_logger.log_value("train blind SINGLE_OBJECT_WITHIN_SCENE", blind_non_match_loss.item(), loss_current_iteration)

            if data_type == SpartanDatasetDataType.DIFFERENT_OBJECT:
                self._tensorboard_logger.log_value("train blind DIFFERENT_OBJECT", blind_non_match_loss.item(), loss_current_iteration)

        # loss is never zero
        self.logging_dict['train']['loss'].append(loss.item())
        if data_type == SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE:
            self._tensorboard_logger.log_value("train loss SINGLE_OBJECT_WITHIN_SCENE", loss.item(), loss_current_iteration)
        elif data_type == SpartanDatasetDataType.DIFFERENT_OBJECT:
            self._tensorboard_logger.log_value("train loss DIFFERENT_OBJECT", loss.item(), loss_current_iteration)
        elif data_type == SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE:
            self._tensorboard_logger.log_value("train loss SINGLE_OBJECT_ACROSS_SCENE", loss.item(), loss_current_iteration)
        elif data_type == SpartanDatasetDataType.MULTI_OBJECT:
            self._tensorboard_logger.log_value("train loss MULTI_OBJECT", loss.item(), loss_current_iteration)
        elif data_type == SpartanDatasetDataType.SYNTHETIC_MULTI_OBJECT:
            self._tensorboard_logger.log_value("train loss SYNTHETIC_MULTI_OBJECT", loss.item(), loss_current_iteration)
        else:
            raise ValueError("unknown data type")

        if data_type == SpartanDatasetDataType.DIFFERENT_OBJECT:
            self._tensorboard_logger.log_value("train different object", loss.item(), loss_current_iteration)
        
        
    
    def run(self,two_input=False,resume=False, resume_folder=None):
        
        loss_current_iteration=0
        #DCE = DenseCorrespondenceEvaluation
        self.save_configs()
        
        if resume:
            if resume_folder is not None:
                iteration = self.resume(resume_folder)
                if iteration is not None:
                    loss_current_iteration = iteration
                logging.info('training from resume')
            else:
                pass
        else:
            pass
        
        start_iteration = copy.copy(loss_current_iteration)
        
        model = self.model
        model.cuda()
        model.train()
        optimizer = self.optimizer
        batch_size = self.data_loader.batch_size
        loss = match_loss = non_match_loss = 0
        max_num_iterations = self.num_iterations + start_iteration
        
        if not resume:
            self.save_network(model, optimizer, 0)
        
        for  epoch  in range(self.max_epochs):
            
            for i, data in enumerate(self.data_loader,0):
                loss_current_iteration +=1
                start_iter_time = time.time()
                match_type, \
                img_a, img_b, \
                img_a_depth, img_b_depth,\
                matches_a, matches_b, \
                masked_non_matches_a, masked_non_matches_b,\
                background_non_matches_a, background_non_matches_b, \
                blind_non_matches_a, blind_non_matches_b, \
                metadata = data
                
                if (match_type == -1).all():
                    print ("\n empty data, continuing \n")
                    continue
                
                data_type = metadata["type"][0]
                img_a = Variable(img_a.cuda(), requires_grad=False)
                img_b = Variable(img_b.cuda(), requires_grad=False)
                img_a_depth = Variable(img_a_depth.cuda(), requires_grad=False)
                img_b_depth = Variable(img_b_depth.cuda(), requires_grad=False)
                matches_a = Variable(matches_a.cuda().squeeze(0), requires_grad=False)
                matches_b = Variable(matches_b.cuda().squeeze(0), requires_grad=False)
                masked_non_matches_a = Variable(masked_non_matches_a.cuda().squeeze(0), requires_grad=False)
                masked_non_matches_b = Variable(masked_non_matches_b.cuda().squeeze(0), requires_grad=False)
                background_non_matches_a = Variable(background_non_matches_a.cuda().squeeze(0), requires_grad=False)
                background_non_matches_b = Variable(background_non_matches_b.cuda().squeeze(0), requires_grad=False)
                blind_non_matches_a = Variable(blind_non_matches_a.cuda().squeeze(0), requires_grad=False)
                blind_non_matches_b = Variable(blind_non_matches_b.cuda().squeeze(0), requires_grad=False)
                
                optimizer.zero_grad()
                self.adjust_learning_rate(optimizer, loss_current_iteration)
                
                # run both images through the network
                if two_input:
                    img_a_pred = model(img_a, img_a_depth/1000.0)
                    img_a_pred = self.process_net_output(img_a_pred, batch_size)
                    img_b_pred = model(img_b, img_b_depth/1000.0)
                    img_b_pred = self.process_net_output(img_b_pred, batch_size)
                else:
                    img_a_pred = model(img_a)
                    img_a_pred = self.process_net_output(img_a_pred, batch_size)
                    img_b_pred = model(img_b)
                    img_b_pred = self.process_net_output(img_b_pred, batch_size)
                    
                
                # get loss
                loss, match_loss, masked_non_match_loss, \
                background_non_match_loss, blind_non_match_loss = get_loss(self.loss_func, match_type,
                                                                                img_a_pred, img_b_pred,
                                                                                matches_a,     matches_b,
                                                                                masked_non_matches_a, masked_non_matches_b,
                                                                                background_non_matches_a, background_non_matches_b,
                                                                                blind_non_matches_a, blind_non_matches_b)
                loss.backward()
                optimizer.step()
                
                elapsed = time.time() -start_iter_time
                
                learning_rate = self.get_learning_rate(optimizer)
                self.update_log(loss_current_iteration,learning_rate, loss, match_loss, \
                                                  masked_non_match_loss, background_non_match_loss, blind_non_match_loss, data_type)
                # save the model
                if loss_current_iteration % self.save_rate == 0:
                    self.save_network(model, optimizer, loss_current_iteration, logging_dict = self.logging_dict)
                
                if loss_current_iteration % self.logging_rate == 0:
                    logging.info("Training on iteration %d of %d" %(loss_current_iteration, max_num_iterations))
                    logging.info("single iteration took %.3f seconds" %(elapsed))
                    percent_complete = loss_current_iteration * 100.0/(max_num_iterations - start_iteration)
                    logging.info("Training is %d percent complete\n" %(percent_complete))
                    print("Training on iter %d of %d , %d precent complete\n"  %(loss_current_iteration,max_num_iterations,percent_complete))
                
                                # don't compute the test loss on the first few times through the loop
                if self.cfg["training"]["compute_test_loss"] and (loss_current_iteration % self.compute_test_loss_rate == 0) and loss_current_iteration > 5:
                    logging.info("Computing test loss")
                    # delete the loss, match_loss, non_match_loss variables so that
                    # pytorch can use that GPU memory
                    del loss, match_loss, masked_non_match_loss, background_non_match_loss, blind_non_match_loss
                    gc.collect()
                    model.eval()
                    test_loss, test_match_loss, test_non_match_loss = DCE.compute_loss_on_dataset(model,
                                                                                                  self._data_loader_test, self.cfg['loss_function'], num_iterations=self.cfg['training']['test_loss_num_iterations'])

                    # delete these variables so we can free GPU memory
                    del test_loss, test_match_loss, test_non_match_loss

                    # make sure to set the network back to train mode
                    model.train()  
                    
                if loss_current_iteration % self.cfg['training']['garbage_collect_rate'] == 0:
                    logging.debug("running garbage collection")
                    gc_start = time.time()
                    gc.collect()
                    gc_elapsed = time.time() - gc_start
                    logging.debug("garbage collection took %.2d seconds" %(gc_elapsed))
                
                if loss_current_iteration > max_num_iterations:
                    logging.info("Finished testing after %d iterations" % (max_num_iterations))
                    self.save_network(model, optimizer, loss_current_iteration, logging_dict=self.logging_dict)
                    return
                       
                



            
                
                
                
                
            
            
            
    
    
    
    
    
    

    
    


