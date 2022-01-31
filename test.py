# import enum
# import torch
# # import torchvision.transforms as gdata
# import os
# import cv2
# # from torchvision.transforms import transforms
# # import matplotlib.pyplot as plt
# import numpy as np
# import myutils

# from Network.model import resnet50


# x = [[ 0.,  1.,  2.,  3.,  4.],
# [ 5.,  6.,  7.,  8.,  9.]]

# x = np.array(x)
# print(type(x))

# # x = torch.tensor(x)
# # y = torch.flip(x,[1])


# # root=os.path.join('~', 'data', 'VOCdevkit')
# # root = os.path.join(os.path.expanduser(root), 'VOC2012')
# # image_path = os.path.join(root, 'JPEGImages', '{}.jpg')
# # img_id = '2007_000033'
# # img = cv2.imread(image_path.format(img_id),cv2.IMREAD_COLOR)
# # cv2.imshow('img',img)
# # print(img)
# # transform = gdata.Compose([gdata.ToTensor(),gdata.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])
# # img2 = transform(img)
# # print(img2)
# # # cv2.imshow('img2',img2)
# # plt.imshow(img2.permute(1, 2, 0))
# # cv2.waitKey()

# # print(y)



# def train_model(model, device, optimizer, scheduler, train_data, val_data, save_dir, model_num, epochs, log_file):
#     tr_loss, tr_map = [], []
#     val_loss, val_map = [], []
#     best_val_map = 0.0
    
#     # Each epoch has a training and validation phase
#     for epoch in range(epochs):
#         print("-------Epoch {}----------".format(epoch+1))
#         log_file.write("Epoch {} >>".format(epoch+1))
#         scheduler.step()
        
#         running_loss = 0.0
#         running_ap = 0.0
        
#         criterion = torch.nn.CrossEntropyLoss(reduction='sum')
#         m = torch.nn.Sigmoid()
        
#         model.train(True)  # Set model to training mode
        
#         for i, batch in enumerate(train_data):
#             #print(data)
#             # target = target.float()
#             for data, label,box in zip(*batch):
#                 gt_label = label[:, :, 4:5].squeeze(axis=-1)
#                 gt_box = label[:, :, :4]
#                 data, gt_label = data.to(device), gt_label.to(device)
#                 optimizer.zero_grad()
#                 cls_pred = model(data)
#                 loss = criterion(cls_pred, gt_label)
#                 running_loss += loss # sum up batch loss
#                 running_ap += myutils.get_ap_score(torch.Tensor.cpu(gt_label).detach().numpy(), torch.Tensor.cpu(m(cls_pred)).detach().numpy()) 
#                 loss.backward()
#                 optimizer.step()
                
#                 #clear variables
#                 # del data, target, output
#                 # gc.collect()
#                 # torch.cuda.empty_cache()
            
#             # zero the parameter gradients
            
#             # Get metrics here
        
#             # Backpropagate the system the determine the gradients
            
#             # Update the paramteres of the model
    
            
            
#             #print("loss = ", running_loss)
            
#         num_samples = float(len(train_data))
#         tr_loss_ = running_loss.item()/num_samples
#         tr_map_ = running_ap/num_samples
        
#         print('train_loss: {:.4f}, train_avg_precision:{:.3f}'.format(
#             tr_loss_, tr_map_))
        
#         log_file.write('train_loss: {:.4f}, train_avg_precision:{:.3f}, '.format(
#             tr_loss_, tr_map_))
        
#         # Append the values to global arrays
#         tr_loss.append(tr_loss_), tr_map.append(tr_map_)
                    
#     return ([tr_loss, tr_map], [val_loss, val_map])
