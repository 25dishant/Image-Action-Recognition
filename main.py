from os import name
import argparse
import pdb
from tabnanny import check
from matplotlib.pyplot import axis
from numpy import mod
#from parso import parse
import numpy as np
# from cv2 import transform
import torch
import logging
import os
import time
from myutils import encode_labels,plot_history
from Network.train import train_model
# from Network.training import train_model
# from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from Loader import metric,voc_fetcher
# from Loader.metric import Loss,RCNNAccMetric
# import Loader.voc_fetcher as voc_fetcher
from Loader.transformations import HORelationDefaultTrainTransform,HORelationDefaultValTransform
import torchvision
# import torchvision.models.resnet as resnet
from Network import network_res_lstm



lr = [3e-4, 1e-2]

# self._root = os.path.join(os.path.expanduser(root), 'VOC2012')



num = 12

if __name__ == '__main__':
    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    # use_cuda = 0
    np.random.seed(1)
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")


    model_dir = os.path.join(os.path.expanduser("~"), "modelsss")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)


    # Dataloader
    train_dataset = voc_fetcher.VOCAction(split='train', augment_box= False, load_box=True,transform=HORelationDefaultTrainTransform(600,1000))
    val_dataset = voc_fetcher.VOCAction(split='val', load_box=True,transform=HORelationDefaultValTransform(600,1000))
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True,drop_last=True)
    val_data = DataLoader(val_dataset, batch_size=1, shuffle=False,drop_last=False)
    eval_metric = metric.VOCMultiClsMApMetric(class_names=val_dataset.classes, ignore_label=-1, voc_action_type=True)
    



    #Model
    net = network_res_lstm.Custom_Model(pretrained=False,device=device).to(device)
    optimizer = torch.optim.SGD([   
            {'params': list(net.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9, 'weight_decay': 1e-5},
            {'params': list(net.parameters())[-1], 'lr': lr[1], 'momentum': 0.9, 'weight_decay': 1e-5}
            ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)


    #Load the weights
    weights_file_path =  os.path.join(model_dir, "model-{}.pth".format(num))
    if os.path.isfile(weights_file_path):
        print("Loading best weights")
        net.load_state_dict(torch.load(weights_file_path))


    #Training
    log_file = open(os.path.join(model_dir, "log-{}.txt".format(num)), "w+")
    log_file.write("----------Experiment {}-----------\n".format(num))
    trn_hist, val_hist = train_model(net, device, optimizer, scheduler, train_data, val_data, model_dir, num, 10, log_file, eval_metric)
    print("Training History: ",trn_hist)
    print("\n")
    print("Validation History: ",val_hist)
    torch.cuda.empty_cache()


    plot_history(trn_hist[0], val_hist[0], "Loss", os.path.join(model_dir, "loss-{}".format(num)))
    plot_history(trn_hist[1], val_hist[1], "Accuracy", os.path.join(model_dir, "accuracy-{}".format(num)))
    log_file.close()