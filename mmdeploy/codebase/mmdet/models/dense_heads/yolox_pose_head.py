# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.utils import filter_scores_and_topk
from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops import multiclass_nms
from mmdeploy.utils import Backend, get_backend


@FUNCTION_REWRITER.register_rewriter(func_name='models.yolox_pose_head.'
                                     'YOLOXPoseHead.predict_by_feat')
def yoloxpose_head__predict_by_feat(self,
                                    cls_scores: List[Tensor],
                                    bbox_preds: List[Tensor],
                                    objectnesses: Optional[List[Tensor]] = None,
                                    kpt_preds: Optional[List[Tensor]] = None,
                                    vis_preds: Optional[List[Tensor]] = None,
                                    batch_img_metas: Optional[List[dict]] = None,
                                    cfg: Optional[ConfigDict] = None,
                                    rescale: bool = True,
                                    with_nms: bool = True) -> List[InstanceData]:
    """Transform a batch of output features extracted by the head into bbox
        and keypoint results.

    Args:
        ctx: Context that contains original meta information.
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        objectnesses (list[Tensor], Optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, 1, H, W).
        batch_img_metas (list[dict], Optional): Batch image meta info.
            Defaults to None.
        cfg (ConfigDict, optional): Test / postprocessing
            configuration, if None, test_cfg would be used.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.

    Returns:
        tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
            where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
            size and the score between 0 and 1. The shape of the second
            tensor in the tuple is (N, num_box), and each element
          
    """

    ctx = FUNCTION_REWRITER.get_context()

    assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        
    deploy_cfg = ctx.cfg
    dtype = cls_scores[0].dtype
    device = cls_scores[0].device
    batch_size = bbox_preds[0].shape[0]
        
    cfg = self.test_cfg if cfg is None else cfg
    
    multi_label = cfg.multi_label
    multi_label &= self.num_classes > 1
    cfg.multi_label = multi_label

    batch_size = cls_scores[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

    self.mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, dtype=dtype, device=device)

    flatten_priors = torch.cat(self.mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full((featmap_size.numel(), ), stride)
        for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
    ]
    flatten_stride = torch.cat(mlvl_strides)

    # flatten cls_scores, bbox_preds and objectness
    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                              self.cls_out_channels)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_objectness = [
        objectness.permute(0, 2, 3, 1).reshape(batch_size, -1)
        for objectness in objectnesses
    ]
    flatten_vis_preds = [
        vis_pred.permute(0, 2, 3, 1).reshape(
            batch_size, -1, self.num_keypoints) for vis_pred in vis_preds
    ]
    flatten_kpt_preds = [
        kpt_pred.permute(0, 2, 3, 1).reshape(
            batch_size, -1, self.num_keypoints * 2) for kpt_pred in kpt_preds
    ]
    flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
    flatten_kpt_preds = torch.cat(flatten_kpt_preds, dim=1)
    flatten_vis_preds = torch.cat(flatten_vis_preds, dim=1).sigmoid()
    
    featmap_sizes = [vis_pred.shape[2:] for vis_pred in vis_preds]
    
    bboxes = self.bbox_coder.decode(
        flatten_priors[None], flatten_bbox_preds, flatten_stride)
    flatten_decoded_kpts = self.decode_pose(flatten_priors, flatten_kpt_preds, flatten_stride)

    scores = flatten_cls_scores * flatten_objectness

    pred_kpts = torch.cat([flatten_decoded_kpts,
                           flatten_vis_preds.unsqueeze(3)],
                          dim=3)
    if not with_nms:
        dets = torch.cat([bboxes, scores], dim=2)
        return dets, pred_kpts

    # backend = get_backend(deploy_cfg)
    # if backend == Backend.TENSORRT:
    #     # pad for batched_nms because its output index is filled with -1
    #     bboxes = torch.cat(
    #         [bboxes,
    #          bboxes.new_zeros((bboxes.shape[0], 1, bboxes.shape[2]))],
    #         dim=1)
    #     scores = torch.cat(
    #         [scores, scores.new_zeros((scores.shape[0], 1, 1))], dim=1)
    #     pred_kpts = torch.cat([
    #         pred_kpts,
    #         pred_kpts.new_zeros((pred_kpts.shape[0], 1, pred_kpts.shape[2],
    #                              pred_kpts.shape[3]))
    #     ],
    #                           dim=1)

    # nms
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.get('nms_thr', post_params.iou_threshold)
    score_threshold = cfg.get('score_threshold', post_params.score_thresholdeshold)
    pre_top_k = post_params.get('pre_top_k', 1000)
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    # do nms
    # _, _, nms_indices = multiclass_nms(
    #     bboxes,
    #     scores,
    #     max_output_boxes_per_class,
    #     iou_threshold,
    #     score_thresholdeshold,
    #     pre_top_k=pre_top_k,
    #     keep_top_k=keep_top_k,
    #     output_index=True)
    
    if cfg.multi_label is False:
        scores, labels = scores.max(1, keepdim=True)
        scores, _, nms_indices, results = filter_scores_and_topk(
            scores,
            score_threshold,
            pre_top_k,
            results=dict(labels=labels[:, 0]))
        labels = results['labels']
    else:
        scores, labels, nms_indices, _ = filter_scores_and_topk(
            scores, score_threshold, pre_top_k)

    batch_inds = torch.arange(batch_size, device=scores.device).view(-1, 1)
    dets = torch.cat([bboxes, scores], dim=2)
    dets = dets[batch_inds, nms_indices, ...]
    pred_kpts = pred_kpts[batch_inds, nms_indices, ...]

    return dets, pred_kpts
