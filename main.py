from os import name
import argparse
from matplotlib.pyplot import axis
from numpy import mod
from parso import parse
# from cv2 import transform
import torch
import logging
import os
import time
from Network.training import train_model
# from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from Loader import metric,voc_fetcher
# from Loader.metric import Loss,RCNNAccMetric
# import Loader.voc_fetcher as voc_fetcher
from Loader.transformations import HORelationDefaultTrainTransform,HORelationDefaultValTransform
import torchvision
# import torchvision.models.resnet as resnet
from Network import model,lr_schedule






if __name__ == '__main__':
    torch.manual_seed(1)
    
    # Dataloader
    train_dataset = voc_fetcher.VOCAction(split='train', augment_box=True, load_box=True,transform=HORelationDefaultTrainTransform(600,1000))
    val_dataset = voc_fetcher.VOCAction(split='val', load_box=True,transform=HORelationDefaultValTransform(600,1000))
    eval_metric = metric.VOCMultiClsMApMetric(class_names=val_dataset.classes, ignore_label=-1, voc_action_type=True)
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True,drop_last=True)
    val_data = DataLoader(val_dataset, batch_size=1, shuffle=False,drop_last=False)
    

    # Network
    # bbox = {
    #     'gt_box':train_data,
    #     'obj_box':None
    # }

    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print("Available device = ", device)
    resnet50 = model.resnet50(pretrained=False)
    # model = model_collections_dict[model_name]
    # model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    resnet50.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    # model.load_state_dict(model_zoo.load_url(model_urls[model_name]))
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    model.to(device)

    lr = [3e-5,1e-6]

    optimizer = torch.optim.SGD([   
            {'params': list(model.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
            {'params': list(model.parameters())[-1], 'lr': lr[1], 'momentum': 0.9}
            ])


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)

    # train_model(resnet50,'cpu',)
    
    # num_filters = resnet50.fc.in_features
    # print(train_dataset._items)
    # print(train_dataset._label_cache)
    # print(val_metric.output_names)
    # print(val_metric)
    # print(val_metric)
    # print(len(train_data))
    # Train(resnet50,train_data,val_data,eval_metric,args)
    for batch in train_data:
        print("Data:",batch[0])
        print('\n')
        print("label:",batch[1])
        print('\n')
        print("box:",batch[2])
        print('\n')
        break