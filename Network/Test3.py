"""Human-object Relation Network."""
from __future__ import absolute_import

import os
import mxnet as mx
# from mxnet.gluon import nn
from torch import nn
from .base import HORelationBase
from libgcv.model.ho_relation.module import HumanObjectRelationModule


class HORelationNet(HORelationBase):
    def __init__(self, features, top_features, classes,
                 short=600, max_size=1000, train_patterns=None,
                 nms_thresh=0.3, nms_topk=400, post_nms=100,
                 roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
                 rpn_channel=1024, base_size=16, scales=(8, 16, 32),
                 ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
                 rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
                 rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
                 num_sample=10, pos_iou_thresh=0.5, num_ctx_per_sample=10,
                 ctx_iou_lb=0.2, ctx_iou_ub=0.75, additional_output=False, global_avg_pool=True,
                 **kwargs):
        super(HORelationNet, self).__init__(
            features=features, top_features=top_features, classes=classes,
            short=short, max_size=max_size, train_patterns=train_patterns,
            nms_thresh=nms_thresh, nms_topk=nms_topk, post_nms=post_nms,
            roi_mode=roi_mode, roi_size=roi_size, stride=stride, clip=clip,
            global_avg_pool=global_avg_pool, **kwargs)
        self._max_batch = 1  # currently only support batch size = 1
        self._num_sample = num_sample
        self._num_ctx_per_sample = num_ctx_per_sample
        self._rpn_test_post_nms = rpn_test_post_nms
        # Use {} to warp non HybridBlock
        self._additional_output = additional_output

        with self.name_scope():
            self.fc = nn.Dense(1024, activation='relu', weight_initializer=mx.init.Normal(0.01))
            self.fc = nn.Linear(1024, activation='relu', weight_initializer=mx.init.Normal(0.01))
            self.fc_ctx = nn.Dense(1024, activation='relu', weight_initializer=mx.init.Normal(0.01))
            self.relation = HumanObjectRelationModule(num_feat=1024, num_group=16, additional_output=additional_output)
            self.class_predictor = nn.Dense(
                self.num_class, weight_initializer=mx.init.Normal(0.01))
            self.ctx_class_predictor = nn.Dense(
                self.num_class, weight_initializer=mx.init.Normal(0.01))

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, gt_box=None, obj_box=None):
        feat = self.features(x)
        rsn_box = obj_box.reshape((-1, 4))

        # create batchid
        rsn_batchid = F.zeros_like(rsn_box.slice_axis(axis=-1, begin=0, end=1))
        rsn_rois = F.concat(*[rsn_batchid, rsn_box], dim=-1)
        gt_batchid = F.zeros_like(gt_box.slice_axis(axis=-1, begin=0, end=1))
        gt_rois = F.concat(*[gt_batchid.reshape((-1, 1)), gt_box.reshape((-1, 4))], dim=-1)

        # ROI features
        if self._roi_mode == 'pool':
            pooled_feat = F.ROIPooling(feat, gt_rois, self._roi_size, 1. / self._stride)
            pooled_ctx_feat = F.ROIPooling(feat, rsn_rois, self._roi_size, 1. / self._stride)
        elif self._roi_mode == 'align':
            pooled_feat = F.contrib.ROIAlign(feat, gt_rois, self._roi_size, 1. / self._stride,
                                             sample_ratio=2)
            pooled_ctx_feat = F.contrib.ROIAlign(feat, rsn_rois, self._roi_size, 1. / self._stride,
                                                 sample_ratio=2)
        else:
            raise ValueError("Invalid roi mode: {}".format(self._roi_mode))

        # RCNN prediction
        top_feat = self.top_features(pooled_feat)
        # contextual region prediction
        top_ctx_feat = self.top_features(pooled_ctx_feat)

        if self.use_global_avg_pool:
            top_feat = self.global_avg_pool(top_feat)
            top_ctx_feat = self.global_avg_pool(top_ctx_feat)

        top_feat = self.fc(top_feat)
        top_ctx_feat = self.fc_ctx(top_ctx_feat)
        if self._additional_output:
            relation_feat, relation_ctx_feat, relation = \
                self.relation(top_feat, top_ctx_feat, gt_box.reshape((-1, 4)), rsn_box)
        else:
            relation_feat, relation_ctx_feat = \
                self.relation(top_feat, top_ctx_feat, gt_box.reshape((-1, 4)), rsn_box)
        top_feat = top_feat + relation_feat
        top_ctx_feat = top_ctx_feat + relation_ctx_feat

        cls_pred = self.class_predictor(top_feat)
        ctx_cls_pred = self.ctx_class_predictor(top_ctx_feat)

        # cls_pred (B * N, C) -> (B, N, C)
        cls_pred = cls_pred.reshape((self._max_batch, -1, self.num_class))
        ctx_cls_pred = ctx_cls_pred.reshape((self._max_batch, -1, self.num_class))

        ctx_cls_pred = ctx_cls_pred.max(axis=1, keepdims=True)
        cls_pred = F.broadcast_add(cls_pred, ctx_cls_pred)

        if self._additional_output:
            return cls_pred, relation
        return cls_pred


def get_horelation(name, dataset, pretrained=False, params='', ctx=mx.cpu(),
                   root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    net = HORelationNet(**kwargs)
    # if pretrained:
    #     if params.strip():
    #         net.load_parameters(params.strip())
    #     else:
    #         from ..model_store import get_model_file
    #         full_name = '_'.join(('horelation', name, dataset))
    #         net.load_parameters(get_model_file(full_name, root=root), ctx=ctx)
    return net


def horelation_resnet50_v1d_voca(pretrained=False, pretrained_base=True, transfer=None, params='', **kwargs):
    if transfer is None:
        # from ..resnetv1b import resnet50_v1d
        from model import resnet50
        from ...data import VOCAction
        classes = VOCAction.CLASSES
        pretrained_base = False if pretrained else pretrained_base
        # base_network = resnet50_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=True)
        base_network = resnet50(pretrained=pretrained_base, progress=False)
        features = nn.Sequential()
        top_features = nn.Sequential()
        # layers = [nn.Conv1d,nn.BatchNorm1d,nn.ReLU,nn.MaxPool1d,'layer1','layer2','layer3']
        for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
            features.add_module(getattr(base_network, layer))
        # for layer in layers:
        #     features.add_module(getattr(base_network, layer))
        for layer in ['layer4']:
            top_features.add_module(getattr(base_network, layer))
        train_patterns = '|'.join(['.*dense', '.*rsn', '.*relation', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
        return get_horelation(
            name='resnet50_v1d', dataset='voca', pretrained=pretrained,
            features=features, top_features=top_features, classes=classes,
            short=600, max_size=1000, train_patterns=train_patterns,
            nms_thresh=0.3, nms_topk=400, post_nms=100,
            roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
            rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
            ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
            rpn_train_pre_nms=50, rpn_train_post_nms=50,
            rpn_test_pre_nms=50, rpn_test_post_nms=50, rpn_min_size=16,
            num_sample=10, pos_iou_thresh=0.5, num_ctx_per_sample=10,
            ctx_iou_lb=0.2, ctx_iou_ub=0.75,  params=params,
            **kwargs)
    else:
        raise NotImplementedError
