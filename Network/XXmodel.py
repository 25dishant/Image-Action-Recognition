from torch import nn,Block
import os

import torch
# from torch.utils import sh
import math
# import mxnet as mx

class HORelationBase(Block):
    """Human-object Relation Network.

    Parameters
    ----------
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    top_features : gluon.HybridBlock
        Tail feature extractor after feature pooling layer.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    short : int
        Input image short side size.
    max_size : int
        Maximum size of input image long side.
    train_patterns : str
        Matching pattern for trainable parameters.
    nms_thresh : float
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    roi_mode : str
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2
        (height, width) of the ROI region.
    stride : int
        Stride of network features.
    clip: float
        Clip bounding box target to this value.

    Attributes
    ----------
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    num_class : int
        Number of positive categories.
    short : int
        Input image short side size.
    max_size : int
        Maximum size of input image long side.
    train_patterns : str
        Matching pattern for trainable parameters.
    nms_thresh : float
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.

    """
    def __init__(self, features, top_features, classes,
                 short, max_size, train_patterns,
                 nms_thresh, nms_topk, post_nms,
                 roi_mode, roi_size, stride, clip,
                 global_avg_pool=True, **kwargs):
        super(HORelationBase, self).__init__(**kwargs)
        self.classes = classes
        self.num_class = len(classes)
        self.short = short
        self.max_size = max_size
        self.train_patterns = train_patterns
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.use_global_avg_pool = global_avg_pool

        assert self.num_class > 0, "Invalid number of class : {}".format(self.num_class)
        assert roi_mode.lower() in ['align', 'pool'], "Invalid roi_mode: {}".format(roi_mode)
        self._roi_mode = roi_mode.lower()
        assert len(roi_size) == 2, "Require (h, w) as roi_size, given {}".format(roi_size)
        self._roi_size = roi_size
        self._stride = stride

        with self.name_scope():
            self.features = features
            self.top_features = top_features
            self.global_avg_pool = nn.GlobalAvgPool2D()

    def collect_train_params(self, select=None):
        """Collect trainable params.

        This function serves as a help utility function to return only
        trainable parameters if predefined by experienced developer/researcher.
        For example, if cross-device BatchNorm is not enabled, we will definitely
        want to fix BatchNorm statistics to avoid scaling problem because RCNN training
        batch size is usually very small.

        Parameters
        ----------
        select : select : str
            Regular expressions for parameter match pattern

        Returns
        -------
        The selected :py:class:`mxnet.gluon.ParameterDict`

        """
        if select is None:
            return self.collect_params(self.train_patterns)
        return self.collect_params(select)

    def set_nms(self, nms_thresh=0.3, nms_topk=400, post_nms=100):
        """Set NMS parameters to the network.

        .. Note::
            If you are using hybrid mode, make sure you re-hybridize after calling
            ``set_nms``.

        Parameters
        ----------
        nms_thresh : float, default is 0.3.
            Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.

        Returns
        -------
        None

        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    def reset_class(self, classes):
        """Reset class categories and class predictors.

        Parameters
        ----------
        classes : iterable of str
            The new categories. ['apple', 'orange'] for example.

        """
        self._clear_cached_op()
        self.classes = classes
        self.num_class = len(classes)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, width, height):
        """Not implemented yet."""
        raise NotImplementedError


class HumanObjectRelationModule(Block):
    r"""Human-object Relation Module.

    Parameters
    ----------
    num_feat: int, default is 1024
        Dimension number used in fc layers.
    num_group : int, default is 16
        Relation group number.
        dk = num_feat / num_group.
    """
    def __init__(self, num_feat=1024, num_group=16, additional_output=False, **kwargs):
        super(HumanObjectRelationModule, self).__init__(**kwargs)
        self.num_feat = num_feat
        self.num_group = num_group
        self.dim_k = int(num_feat / num_group)
        self.additional_output = additional_output
        # weight_initializer = mx.init.Normal(0.01)
        weight_initializer = torch.nn.init.normal_(std=0.01)
        with self.name_scope():
            self.fc_gt_ctx_position = nn.Dense(self.num_group, activation='relu', weight_initializer=weight_initializer)
            self.fc_ctx_gt_position = nn.Dense(self.num_group, activation='relu', weight_initializer=weight_initializer)
            self.fc_gt = nn.Dense(self.num_feat, weight_initializer=weight_initializer)
            self.fc_ctx = nn.Dense(self.num_feat, weight_initializer=weight_initializer)
            self.gt_ctx_linear_out = nn.Conv2D(self.num_feat, 1, 1, 0, groups=self.num_group,
                                               weight_initializer=weight_initializer)
            self.ctx_gt_linear_out = nn.Conv2D(self.num_feat, 1, 1, 0, groups=self.num_group,
                                               weight_initializer=weight_initializer)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, feat, ctx_feat, box, ctx_box):
        """Forward Relation Module.

        Parameters
        ----------
        feat : mxnet.nd.NDArray or mxnet.symbol
            (M, 1024) Feature tensor (used to compute q).
        ctx_feat : mxnet.nd.NDArray or mxnet.symbol
            (N, 1024)Contextual Feature tensor (used to compute k,v).
        box: mxnet.nd.NDArray or mxnet.symbol
            (M, 4) boxes with corner encoding.
        ctx_box: mxnet.nd.NDArray or mxnet.symbol
            (N, 4) boxes with corner encoding.

        Returns
        -------
        gt_relation_feat, ctx_relation_feat
            (M, 1024).

        """
        gt_ctx_pos_embedding = self.position_embedding(F, box, ctx_box, feat_dim=64)  # (M*N, feat_dim)
        gt_ctx_pos_feat = self.fc_gt_ctx_position(gt_ctx_pos_embedding)  # (M*N, num_group)
        gt_ctx_pos_feat = gt_ctx_pos_feat.transpose()  # (num_group, M*N)

        ctx_gt_pos_embedding = self.position_embedding(F, ctx_box, box, feat_dim=64)  # (N*M, feat_dim)
        ctx_gt_pos_feat = self.fc_ctx_gt_position(ctx_gt_pos_embedding)  # (N*M, num_group)
        ctx_gt_pos_feat = ctx_gt_pos_feat.transpose()  # (num_group, N*M)

        gt_data = self.fc_gt(feat)
        gt_data = gt_data.reshape((-1, self.num_group, self.dim_k)).transpose(axes=(1, 0, 2))  # (num_group, M, dim_k)
        ctx_data = self.fc_ctx(ctx_feat)
        ctx_data = ctx_data.reshape((-1, self.num_group, self.dim_k)).transpose(axes=(1, 0, 2))  # (num_group, N, dim_k)

        gt_ctx = F.batch_dot(lhs=gt_data, rhs=ctx_data, transpose_a=False, transpose_b=True)  # (num_group, M, N)
        gt_ctx = (1.0 / math.sqrt(float(self.dim_k))) * gt_ctx
        ctx_gt = F.transpose(gt_ctx, axes=(0, 2, 1))  # (num_group, N, M)

        gt_ctx_pos_feat = F.reshape_like(gt_ctx_pos_feat, gt_ctx)
        gt_ctx = F.transpose(gt_ctx, axes=(1, 0, 2))  # (M, num_group, N)
        gt_ctx_pos_feat = F.transpose(gt_ctx_pos_feat, axes=(1, 0, 2))  # (M, num_group, N)

        weighted_gt_ctx = F.log(F.maximum(gt_ctx_pos_feat, 1e-6)) + gt_ctx
        weighted_gt_ctx = F.softmax(data=weighted_gt_ctx, axis=2)
        weighted_gt_ctx = weighted_gt_ctx.reshape((-3, -2))  # (M * num_group, N)

        gt_output = F.dot(lhs=weighted_gt_ctx, rhs=ctx_feat)  # (M * num_group, 1024)
        gt_output = gt_output.reshape((-1, self.num_group*self.num_feat, 1, 1))  # (M, num_group*1024, 1, 1)
        gt_output = self.gt_ctx_linear_out(gt_output)  # (M, 1024, 1, 1)

        ctx_gt_pos_feat = F.reshape_like(ctx_gt_pos_feat, ctx_gt)  # (num_group, N, M)
        ctx_gt = F.transpose(ctx_gt, axes=(1, 0, 2))  # (N, num_group, M)
        ctx_gt_pos_feat = F.transpose(ctx_gt_pos_feat, axes=(1, 0, 2))  # (N, num_group, M)

        weighted_ctx_gt = F.log(F.maximum(ctx_gt_pos_feat, 1e-6)) + ctx_gt
        weighted_ctx_gt = F.softmax(data=weighted_ctx_gt, axis=2)
        weighted_ctx_gt = weighted_ctx_gt.reshape((-3, -2))  # (N * num_group, M)

        ctx_output = F.dot(lhs=weighted_ctx_gt, rhs=feat)  # (N * num_group, 1024)
        ctx_output = ctx_output.reshape((-1, self.num_group * self.num_feat, 1, 1))  # (N, num_group*1024, 1, 1)
        ctx_output = self.ctx_gt_linear_out(ctx_output)  # (N, 1024, 1, 1)

        if self.additional_output:
            # (M * num_group, N) -> # (M, num_group, N) -> # (M, N)
            gt_ctx_relation = F.mean(weighted_gt_ctx.reshape(-4, -1, self.num_group, -2), axis=1, keepdims=False)
            return gt_output.reshape((0, 0)), ctx_output.reshape((0, 0)), gt_ctx_relation

        return gt_output.reshape((0, 0)), ctx_output.reshape((0, 0))

    def position_embedding(self, F, box, ctx_box, feat_dim=64, wave_length=1000):
        """Compute position embedding.

        Parameters
        ----------
        box: mxnet.nd.NDArray or mxnet.symbol
            (M, 4) boxes with corner encoding.
        ctx_box: mxnet.nd.NDArray or mxnet.symbol
            (N, 4) boxes with corner encoding.
        feat_dim: int, default is 64
        wave_length: int default is 1000

        Returns
        -------
        embedding
            Returns (M, N, feat_dim).
        """
        # position encoding
        # (M, 1)
        xmin, ymin, xmax, ymax = F.split(data=box, num_outputs=4, axis=1)
        box_width = xmax - xmin + 1.
        box_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        # (N, 1)
        ctx_xmin, ctx_ymin, ctx_xmax, ctx_ymax = F.split(data=ctx_box, num_outputs=4, axis=1)
        ctx_box_width = ctx_xmax - ctx_xmin + 1.
        ctx_box_height = ctx_ymax - ctx_ymin + 1.
        ctx_center_x = 0.5 * (ctx_xmin + ctx_xmax)
        ctx_center_y = 0.5 * (ctx_ymin + ctx_ymax)

        # (M, N)
        delta_x = F.broadcast_minus(lhs=center_x, rhs=F.transpose(ctx_center_x))
        delta_x = F.broadcast_div(delta_x, box_width)
        delta_x = F.log(F.maximum(F.abs(delta_x), 1e-3))
        delta_y = F.broadcast_minus(lhs=center_y, rhs=F.transpose(ctx_center_y))
        delta_y = F.broadcast_div(delta_y, box_height)
        delta_y = F.log(F.maximum(F.abs(delta_y), 1e-3))
        delta_width = F.broadcast_div(lhs=F.transpose(ctx_box_width), rhs=box_width)
        delta_width = F.log(delta_width)
        delta_height = F.broadcast_div(lhs=F.transpose(ctx_box_height), rhs=box_height)
        delta_height = F.log(delta_height)
        # (M, N, 4)
        position_mat = F.stack(*[delta_x, delta_y, delta_width, delta_height], axis=2)

        # position embedding
        feat_range = F.arange(0, feat_dim / 8)
        dim_mat = F.broadcast_power(lhs=F.full((1,), wave_length), rhs=(8. / feat_dim) * feat_range)
        dim_mat = F.Reshape(dim_mat, shape=(1, 1, 1, -1))  # (1, 1, 1, feat_dim/8)
        # position_mat (M, N, 4, 1)
        position_mat = F.expand_dims(100.0 * position_mat, axis=3)
        div_mat = F.broadcast_div(lhs=position_mat, rhs=dim_mat)  # (M, N, 4, feat_dim/8)
        sin_mat = F.sin(data=div_mat)
        cos_mat = F.cos(data=div_mat)
        embedding = F.concat(sin_mat, cos_mat, dim=3)   # (M, N, 4, feat_dim/4)
        return embedding.reshape((-3, feat_dim))



class HORelationNet(HORelationBase):
    r"""Human-object Relation Network.

    Parameters
    ----------
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    top_features : gluon.HybridBlock
        Tail feature extractor after feature pooling layer.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    short : int, default is 600.
        Input image short side size.
    max_size : int, default is 1000.
        Maximum size of input image long side.
    train_patterns : str, default is None.
        Matching pattern for trainable parameters.
    nms_thresh : float, default is 0.3.
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    roi_mode : str, default is align
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2, default is (14, 14)
        (height, width) of the ROI region.
    stride : int, default is 16
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    clip : float, default is None
        Clip bounding box target to this value.
    rpn_channel : int, default is 1024
        Channel number used in RPN convolutional layers.
    base_size : int
        The width(and height) of reference anchor box.
    scales : iterable of float, default is (8, 16, 32)
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}

    ratios : iterable of float, default is (0.5, 1, 2)
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    rpn_train_pre_nms : int, default is 12000
        Filter top proposals before NMS in training of RPN.
    rpn_train_post_nms : int, default is 2000
        Return top proposal results after NMS in training of RPN.
    rpn_test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing of RPN.
    rpn_test_post_nms : int, default is 300
        Return top proposal results after NMS in testing of RPN.
    rpn_nms_thresh : float, default is 0.7
        IOU threshold for NMS. It is used to remove overlapping proposals.
    train_pre_nms : int, default is 12000
        Filter top proposals before NMS in training.
    train_post_nms : int, default is 2000
        Return top proposal results after NMS in training.
    test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing.
    test_post_nms : int, default is 300
        Return top proposal results after NMS in testing.
    rpn_min_size : int, default is 16
        Proposals whose size is smaller than ``min_size`` will be discarded.
    num_sample : int, default is 128
        Number of samples for RCNN targets.
    pos_iou_thresh : float, default is 0.5
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
    pos_ratio : float, default is 0.25
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    additional_output : boolean, default is False
        ``additional_output`` is only used for Mask R-CNN to get internal outputs.
    test_gt_box_input: boolean, default is False
        When true, require and use gt_box as proposal in test forward.

    Attributes
    ----------
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    num_class : int
        Number of positive categories.
    short : int
        Input image short side size.
    max_size : int
        Maximum size of input image long side.
    train_patterns : str
        Matching pattern for trainable parameters.
    nms_thresh : float
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    target_generator : gluon.Block
        Generate training targets with boxes, samples, matches, gt_label and gt_box.

    """
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
            # self.fc = nn.Dense(1024, activation='relu', weight_initializer=mx.init.Normal(0.01))
            self.fc = nn.Dense(1024, activation='relu', weight_initializer=torch.nn.init.normal_(std=0.01))
            self.fc_ctx = nn.Dense(1024, activation='relu', weight_initializer=torch.nn.init.normal_(std=0.01))
            self.relation = HumanObjectRelationModule(num_feat=1024, num_group=16, additional_output=additional_output)
            self.class_predictor = nn.Dense(
                self.num_class, weight_initializer=torch.nn.init.normal_(std=0.01))
            self.ctx_class_predictor = nn.Dense(
                self.num_class, weight_initializer=torch.nn.init.normal_(std=0.01))

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, gt_box=None, obj_box=None):
        """Forward Faster-RCNN network.

        The behavior during traing and inference is different.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            The network input tensor.
        gt_box : mxnet.nd.NDArray or mxnet.symbol
            The ground-truth bbox tensor with shape (1, N, 4).
        obj_box : mxnet.nd.NDArray or mxnet.symbol
            The object bbox tensor with shape (1, N, 4).

        Returns
        -------
        (ids, scores, bboxes)
            During inference, returns final class id, confidence scores, bounding
            boxes.

        """
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


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]


def get_model_file(name, root=os.path.join('~', '.mxnet', 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    file_name = '{name}-{short_hash}'.format(name=name,
                                             short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.params')
    sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            raise AssertionError('Mismatch in the content of model file detected. Please download it again.')
    else:
        raise AssertionError('Model file: %s is not found. Please download before use it.' % file_path)



class BottleneckV1b(Block):
    """ResNetV1b BottleneckV1b
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, planes, strides=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None,
                 norm_kwargs=None, last_gamma=False, **kwargs):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2D(channels=planes, kernel_size=1,
                               use_bias=False)
        self.bn1 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.relu1 = nn.Activation('relu')
        self.conv2 = nn.Conv2D(channels=planes, kernel_size=3, strides=strides,
                               padding=dilation, dilation=dilation, use_bias=False)
        self.bn2 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.relu2 = nn.Activation('relu')
        self.conv3 = nn.Conv2D(channels=planes * 4, kernel_size=1, use_bias=False)
        if not last_gamma:
            self.bn3 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        else:
            self.bn3 = norm_layer(gamma_initializer='zeros',
                                  **({} if norm_kwargs is None else norm_kwargs))
        self.relu3 = nn.Activation('relu')
        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides

    def forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu3(out)

        return out


class ResNetV1b(Block):
    """ Pre-trained ResNetV1b Model, which produces the strides of 8
    featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.


    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, classes=1000, dilated=False, norm_layer=nn.BatchNorm2d,
                 norm_kwargs=None, last_gamma=False, deep_stem=False, stem_width=32,
                 avg_down=False, final_drop=0.0, use_global_stats=False,
                 name_prefix='', **kwargs):
        self.inplanes = stem_width*2 if deep_stem else 64
        super(ResNetV1b, self).__init__(prefix=name_prefix)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['use_global_stats'] = True
        self.norm_kwargs = norm_kwargs
        with self.name_scope():
            if not deep_stem:
                self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                       padding=3, use_bias=False)
            else:
                self.conv1 = nn.HybridSequential(prefix='conv1')
                self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                         padding=1, use_bias=False))
                self.conv1.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.conv1.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
            self.bn1 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.layer1 = self._make_layer(1, block, 64, layers[0], avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma)
            self.layer2 = self._make_layer(2, block, 128, layers[1], strides=2, avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma)
            if dilated:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=4,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma)
            else:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma)
            self.avgpool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.drop = None
            if final_drop > 0.0:
                self.drop = nn.Dropout(final_drop)
            self.fc = nn.Dense(in_units=512 * block.expansion, units=classes)

    def _make_layer(self, stage_index, block, planes, blocks, strides=1, dilation=1,
                    avg_down=False, norm_layer=None, last_gamma=False):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix='down%d_'%stage_index)
            with downsample.name_scope():
                if avg_down:
                    if dilation == 1:
                        downsample.add(nn.AvgPool2D(pool_size=strides, strides=strides,
                                                    ceil_mode=True, count_include_pad=False))
                    else:
                        downsample.add(nn.AvgPool2D(pool_size=1, strides=1,
                                                    ceil_mode=True, count_include_pad=False))
                    downsample.add(nn.Conv2D(channels=planes * block.expansion, kernel_size=1,
                                             strides=1, use_bias=False))
                    downsample.add(norm_layer(**self.norm_kwargs))
                else:
                    downsample.add(nn.Conv2D(channels=planes * block.expansion,
                                             kernel_size=1, strides=strides, use_bias=False))
                    downsample.add(norm_layer(**self.norm_kwargs))

        layers = nn.HybridSequential(prefix='layers%d_'%stage_index)
        with layers.name_scope():
            if dilation in (1, 2):
                layers.add(block(planes, strides, dilation=1,
                                 downsample=downsample, previous_dilation=dilation,
                                 norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                                 last_gamma=last_gamma))
            elif dilation == 4:
                layers.add(block(planes, strides, dilation=2,
                                 downsample=downsample, previous_dilation=dilation,
                                 norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                                 last_gamma=last_gamma))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))

            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.add(block(planes, dilation=dilation,
                                 previous_dilation=dilation, norm_layer=norm_layer,
                                 norm_kwargs=self.norm_kwargs, last_gamma=last_gamma))

        return layers

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x


# def get_horelation(name, dataset, pretrained=False, params='', ctx=torch.Tensor.cpu(0),
#                    root=os.path.join('~', '.mxnet', 'models'), **kwargs):
def get_horelation(name, dataset, pretrained=False, params='',
                   root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Utility function to return a network.

    Parameters
    ----------
    name : str
        Model name.
    dataset : str
        The name of dataset.
    pretrained : bool, optional, default is False
        Load pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The Faster-RCNN network.

    """
    net = HORelationNet(**kwargs)
    if pretrained:
        if params.strip():
            net.load_parameters(params.strip())
        else:
            # from ..model_store import get_model_file
            full_name = '_'.join(('horelation', name, dataset))
            net.load_parameters(get_model_file(full_name, root=root), ctx=ctx)
    return net

# def resnet50_v1d(pretrained=False, root='~/.mxnet/models', ctx=torch.Tensor.cpu(), name_prefix='resnetv1d_', **kwargs):
def resnet50_v1d(pretrained=False, root='~/.mxnet/models', name_prefix='resnetv1d_', **kwargs):
    """Constructs a ResNetV1d-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`).
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, avg_down=True,
                      name_prefix=name_prefix, **kwargs)
    if pretrained:
        # from .model_store import get_model_file
        model.load_parameters(get_model_file('resnet%d_v%dd'%(50, 1),
                                             root=root), ctx=ctx)
    return model


def horelation_resnet50_v1d_voca(pretrained=False, pretrained_base=True, transfer=None, params='', **kwargs):
    r"""Human-object Relation Model

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from faster RCNN networks trained
        on other datasets.
    params : str
        If not '', will load prams file form this path.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    """
    if transfer is None:
        # from ..resnetv1b import resnet50_v1d
        # from ..Loader.voc_fetcher import VOCAction
        # classes = VOCAction.CLASSES
        classes = ('jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike',
               'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking', 'other')
        pretrained_base = False if pretrained else pretrained_base
        base_network = resnet50_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=True)
        # features = nn.Sequential(nn.Conv1d,nn.BatchNorm1d,nn.ReLU,nn.MaxPool1d)
        features = nn.Sequential()
        top_features = nn.Sequential()
        for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
            features.add_module(getattr(base_network,layer))
            # features.add(getattr(base_network, layer))
        for layer in ['layer4']:
            top_features.add_module(getattr(base_network,layer))
            # top_features.add(getattr(base_network, layer))
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