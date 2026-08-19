import copy
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmdet.structures import DetDataSample, SampleList
from mmengine.config import Config
from mmengine.dataset import default_collate
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmdet.registry import DATASETS
from mmdet.utils import ConfigType, get_test_pipeline_cfg

# from ..evaluation import get_classes
# from ..registry import MODELS
# from ..structures import DetDataSample, SampleList
# from ..utils import get_test_pipeline_cfg


def custom_collate(batch):
    """Custom collate function to merge list of dictionaries into a batch."""
    collated = {
        'inputs': default_collate([item['inputs'] for item in batch]),
        'data_samples': default_collate([item['data_samples'] for item in batch])
    }
    return collated


ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def inference_detector(
    model: nn.Module,
    imgs: ImagesType,
    test_pipeline: Optional[Compose] = None,
) -> Union[DetDataSample, SampleList]:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    scope = "mmpose"
    if scope is not None:
        init_default_scope(scope)

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

        test_pipeline = Compose(test_pipeline)

    result_list = []
    batch_data = {"inputs": [], "data_samples": []}  # for batch operation
    for i, img in enumerate(imgs):
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)

        # build the data pipeline
        data_ = test_pipeline(data_)

        # data_['inputs'] = [data_['inputs']]
        # data_['data_samples'] = [data_['data_samples']]

        # batch operation
        batch_data['inputs'].append(data_['inputs'])
        batch_data['data_samples'].append(data_['data_samples'])

    # forward the model
    with torch.no_grad():
        # results = model.test_step(data_)[0]
        results = model.test_step(batch_data)

        # result_list.append(results)
    return results

    # if not is_batch:
    #     return result_list[0]
    # else:
    #     return result_list