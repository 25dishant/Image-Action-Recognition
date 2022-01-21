# import torch
# import torchvision.transforms as gdata
# import os
# import cv2
# from torchvision.transforms import transforms
# import matplotlib.pyplot as plt
import numpy as np


x = [[ 0.,  1.,  2.,  3.,  4.],
[ 5.,  6.,  7.,  8.,  9.]]

x = np.array(x)
print(type(x))

# x = torch.tensor(x)
# y = torch.flip(x,[1])


# root=os.path.join('~', 'data', 'VOCdevkit')
# root = os.path.join(os.path.expanduser(root), 'VOC2012')
# image_path = os.path.join(root, 'JPEGImages', '{}.jpg')
# img_id = '2007_000033'
# img = cv2.imread(image_path.format(img_id),cv2.IMREAD_COLOR)
# cv2.imshow('img',img)
# print(img)
# transform = gdata.Compose([gdata.ToTensor(),gdata.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])
# img2 = transform(img)
# print(img2)
# # cv2.imshow('img2',img2)
# plt.imshow(img2.permute(1, 2, 0))
# cv2.waitKey()





# print(y)