import os

import numpy as np
from click.core import batch
from mmengine.dataset import default_collate

from . import util
import cv2
import mmcv
import torch
import matplotlib.pyplot as plt

from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.structures import merge_data_samples

from mmdet.apis import init_detector
# from mmdet.apis import init_detector, inference_detector
from DWPose_usage.inference_detector import inference_detector

from DWPose_usage.inference_topdown import inference_topdown
# from mmpose.apis import inference_topdown


class Wholebody:
    def __init__(self):
        device = 'cuda:0'
        det_config = 'DWPose_usage/yolox_config/yolox_l_8xb8-300e_coco.py'
        det_ckpt = 'DWPose_usage/yolox_config/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'  # https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
        pose_config = 'DWPose_usage/dwpose_config/dwpose-l_384x288.py'
        pose_ckpt = 'DWPose_usage/dwpose_config/dw-ll_ucoco_384.pth'

        # build detector
        self.detector = init_detector(det_config, det_ckpt, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        # build pose estimator
        self.pose_estimator = init_pose_estimator(
            pose_config,
            pose_ckpt,
            device=device)

    def get_bboxes(self, oriImg):
        with torch.no_grad():
            bboxes = []
            bbox_thre = 0.3
            # predict bboxes for a batch of images
            det_result_ = inference_detector(self.detector, oriImg)
            for i, det_result in enumerate(det_result_):
                pred_instance = det_result.pred_instances.cpu().numpy()
                bboxes_i = np.concatenate(
                    (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
                bboxes_i = bboxes_i[np.logical_and(pred_instance.labels == 0,
                                                   pred_instance.scores > bbox_thre)]
                # # max value
                # if len(bboxes) > 0:
                #     bboxes = bboxes[0].reshape(1,-1)
                bboxes_i = bboxes_i[nms(bboxes_i, bbox_thre), :4]

                if bboxes_i.shape == (0, 4):
                    bboxes.append(np.zeros((1, 4), dtype=int))
                    continue

                # Find the largest bounding box
                if bboxes_i.shape != (1, 4):
                    # print(f"bbox shape: {bboxes_i.shape}")
                    # Calculate areas for all bounding boxes
                    areas = (bboxes_i[:, 2] - bboxes_i[:, 0]) * (bboxes_i[:, 3] - bboxes_i[:, 1])
                    # print(f"area: {areas}, bbox shape: {bboxes_i}, {bboxes_i.shape}")
                    # Find the index of the bounding box with the largest area
                    max_area_index = np.argmax(areas)
                    # Select the largest bounding box
                    largest_bbox = bboxes_i[max_area_index].reshape(1, 4)
                    # Append the largest bounding box to the list
                    bboxes.append(largest_bbox)
                    # print(f"largest bbox: {largest_bbox.shape}")
                else:
                    bboxes.append(bboxes_i)
            return bboxes

    def get_pose_sequence(self, oriImg, bboxes):
        with torch.no_grad():
            # predict keypoints
            if len(bboxes) == 0:
                pose_results = inference_topdown(self.pose_estimator, oriImg)
            else:
                pose_results = inference_topdown(self.pose_estimator, oriImg, bboxes)

            keypoints_list = []
            score_list = []

            for i in range(len(pose_results)):
                preds = merge_data_samples([pose_results[i]])
                preds = preds.pred_instances

                # preds = pose_results[0].pred_instances
                keypoints = preds.get('transformed_keypoints',
                                      preds.keypoints)
                if 'keypoint_scores' in preds:
                    scores = preds.keypoint_scores
                else:
                    scores = np.ones(keypoints.shape[:-1])

                if 'keypoints_visible' in preds:
                    visible = preds.keypoints_visible
                else:
                    visible = np.ones(keypoints.shape[:-1])
                keypoints_info = np.concatenate(
                    (keypoints, scores[..., None], visible[..., None]),
                    axis=-1)
                # compute neck joint
                neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
                # neck score when visualizing pred
                neck[:, 2:4] = np.logical_and(
                    keypoints_info[:, 5, 2:4] > 0.3,
                    keypoints_info[:, 6, 2:4] > 0.3).astype(int)
                new_keypoints_info = np.insert(
                    keypoints_info, 17, neck, axis=1)
                mmpose_idx = [
                    17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
                ]
                openpose_idx = [
                    1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
                ]
                new_keypoints_info[:, openpose_idx] = \
                    new_keypoints_info[:, mmpose_idx]
                keypoints_info = new_keypoints_info

                keypoints, scores, visible = keypoints_info[
                                             ..., :2], keypoints_info[..., 2], keypoints_info[..., 3]

                bbox = None
                if len(bboxes) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    bbox = bboxes[i]
                # return keypoints, scores, bbox
                keypoints_list.append(keypoints)
                score_list.append(scores)

            return keypoints_list, score_list
