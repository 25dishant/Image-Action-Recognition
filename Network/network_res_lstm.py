from matplotlib.pyplot import polar
import torch
from zmq import device
# from Network.model import resnet50
from torchvision.models import resnet50
import torchvision
import pdb


class Custom_Model(torch.nn.Module):
    def __init__(self,pretrained,device):
        """
        
        """
        super(Custom_Model, self).__init__()
        self.resnet = resnet50(pretrained=pretrained)
        print("rngrfireniern: ",list(self.resnet.children()))
        self.model1 = torch.nn.Sequential(*(list(self.resnet.children())[0:7]))
        self.model2 = torch.nn.Sequential(*(list(self.resnet.children())[7:9]))
        # self.model2 = torch.nn.Sequential()
        self.model3 = torch.nn.LSTM(1024,11)
        self.device = device

        # self.global_avg_pool = torch.nn.AvgPool2d(kernel_size=(7,7),stride=2)
        self.fc = torch.nn.Linear(in_features= 2048, out_features=1024, bias=True)

        
    def forward(self, x, gt_box = None, obj_box = None):
        """

        """
        pdb.set_trace()
        # obj_box = obj_box.reshape((-1, 4))
        obj_box = list(obj_box)
        gt_box = list(gt_box)
        # print("gt_box: ",gt_box)
        # print("obj_box: ",obj_box)
        fourth_layer_output = self.model1(x)
        # pooled_feat = torchvision.ops.roi_pool(fourth_layer_output, gt_box, output_size=(14, 14), spatial_scale=0.0625)
        # pooled_ctx_feat = torchvision.ops.roi_pool(fourth_layer_output, obj_box, output_size=(14, 14), spatial_scale=0.0625)
        pooled_feat = torchvision.ops.roi_align(fourth_layer_output, gt_box, output_size=(14, 14), spatial_scale=0.0625,sampling_ratio=2)
        pooled_ctx_feat = torchvision.ops.roi_align(fourth_layer_output, obj_box, output_size=(14, 14), spatial_scale=0.0625,sampling_ratio=2)
        # pooled_feat = pooled_feat[0,:,:,:].reshape(1,1024,14,14)
        # pooled_ctx_feat = pooled_ctx_feat[0,:,:,:].reshape(1,1024,14,14)
        top_feat = self.model2(pooled_feat)
        top_ctx_feat = self.model2(pooled_ctx_feat)
        print("CONTROL IS HERE")
        
        # top_feat = self.global_avg_pool(pooled_feat)
        # top_ctx_feat = self.global_avg_pool(pooled_ctx_feat)
        top_feat = top_feat.flatten(1,3)
        top_ctx_feat = top_ctx_feat.flatten(1,3)
        
        top_feat = self.fc(top_feat)
        top_ctx_feat = self.fc(top_ctx_feat)

        hidden = (torch.zeros(1, 1, 11).to(self.device), torch.zeros(1, 1, 11).to(self.device))
        # hidden = hidden.to(self.device)
        for t1 in top_feat:
            t1 = t1.reshape(1,1,1024)
            out1, hidden = self.model3(t1,hidden)
        for t2 in top_ctx_feat:
            t2 = t2.reshape(1,1,1024)
            out, hidden = self.model3(t2,hidden)
            
        print(out)

        return out

