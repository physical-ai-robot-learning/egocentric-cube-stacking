"""Native (non-LeRobot) reader of the processed ego_cart20_v2 tree and its sample weights
(was dataset/xvla_ego_dataset.py + dataset/sampler.py).

XVLAEgoDataset: one sample = one TRAINING row.  Returns state (20, RELCART20) / action (16, 32) / cart20 (16, 20) /
state_prevrel / prev_valid / instruction / order / stage, and optionally images (decoded from the source mp4 at the row's
frame index, full frame resized to `imageSize`).  Shape asserts of spec 27 run on every item.
Sample weights (spec 19/21): order balancing x stage weighting, as weights only -- files are never duplicated and labels
never changed.  stage -1 (unknown) gets weight 1.0.
NOTE: the X-VLA REL-only recipe trains through lerobot_train (uniform sampling over LeRobot frames); these weights are for
the native dataset class / a custom sampler (experiment D) and are not used by the default pretrain.

Pipeline stage: training input (alternative to the LeRobot export).
"""
from __future__ import annotations

import collections
import json
import pathlib

import numpy as np

from .config import CART_DIM, HORIZON, MODEL_ACTION_DIM, STATE_DIM
from .processed_episode import loadEpisode

SAMPLER_BLOCK = 4096        # indices drawn per rng call


def sampleWeights(orders, stages, orderBalanced=True, stageWeights=(1.0, 1.0, 1.0, 2.0)) -> np.ndarray:
    """per-sample weights (sum 1): 1 / (n_orders * count(order)) if orderBalanced, times the stage weight"""
    tOrders = np.asarray(orders)
    tStages = np.asarray(stages, int)
    tW = np.ones(len(tOrders))
    if orderBalanced:
        tCount = collections.Counter(tOrders.tolist())
        tW *= np.array([len(tOrders) / (len(tCount) * tCount[tO]) for tO in tOrders])
    tStageW = np.array([stageWeights[tS] if 0 <= tS < len(stageWeights) else 1.0 for tS in tStages])
    tW *= tStageW
    return tW / tW.sum()


class WeightedSampler:
    """infinite with-replacement index stream (numpy rng; torch-free).  torch: torch.utils.data.WeightedRandomSampler(w, n)"""

    def __init__(self, weights, seed=0):
        self.mWeights = np.asarray(weights)
        self.mRng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            yield from self.mRng.choice(len(self.mWeights), size=SAMPLER_BLOCK, p=self.mWeights).tolist()


class XVLAEgoDataset:
    """<root>/<split>_manifest.jsonl -> one item per training row of every listed episode"""

    def __init__(self, root, split="train", loadImages=False, imageSize=224):
        self.mRoot = pathlib.Path(root)
        self.mLoadImages = loadImages
        self.mImageSize = imageSize
        self.mManifest = [json.loads(tLine) for tLine in open(self.mRoot / f"{split}_manifest.jsonl")]
        self.mEpisodes = [loadEpisode(self.mRoot / tRec["path"]) for tRec in self.mManifest]
        self.mIndex = [(tE, tJ) for tE, tEp in enumerate(self.mEpisodes) for tJ in range(len(tEp["train_rows"]))]

    def __len__(self):
        return len(self.mIndex)

    def weights(self, orderBalanced=True, stageWeights=(1.0, 1.0, 1.0, 2.0)) -> np.ndarray:
        """sampleWeights over every item (stack order of its episode, stage of its row)"""
        tOrders = [self.mEpisodes[tE]["metadata"]["stack_order"] for tE, _ in self.mIndex]
        tStages = [int(self.mEpisodes[tE]["stage_id"][self.mEpisodes[tE]["train_rows"][tJ]]) for tE, tJ in self.mIndex]
        return sampleWeights(tOrders, tStages, orderBalanced, stageWeights)

    def __getitem__(self, i):
        tE, tJ = self.mIndex[i]
        tEp = self.mEpisodes[tE]
        tRow = int(tEp["train_rows"][tJ])
        tMeta = tEp["metadata"]
        tItem = dict(state=np.asarray(tEp["state"][tRow], np.float32), action=np.asarray(tEp["action"][tJ], np.float32),
                     cart20=np.asarray(tEp["cart20"][tJ], np.float32),
                     state_prevrel=np.asarray(tEp["state_prevrel"][tRow], np.float32),
                     state_prev_valid=np.float32(tEp["state_prev_valid"][tRow]), instruction=tMeta["instruction"],
                     stack_order=tMeta["stack_order"], stage_id=int(tEp["stage_id"][tRow]),
                     camera_valid=np.asarray(tEp["camera_valid"][tRow]), episode_id=tMeta["episode_id"], row=tRow)
        assert (tItem["state"].shape == (STATE_DIM,) and tItem["cart20"].shape == (HORIZON, CART_DIM)
                and tItem["action"].shape == (HORIZON, MODEL_ACTION_DIM))
        assert (np.array_equal(tItem["action"][:, :CART_DIM], tItem["cart20"])
                and np.count_nonzero(tItem["action"][:, CART_DIM:]) == 0)
        if self.mLoadImages:
            tItem["images"] = {tCam: self._frame(tEp["cameras"][tCam], tRow)
                               for tCam in tMeta["camera_order"] if tCam in tEp["cameras"]}
        return tItem

    def _frame(self, cam, row):
        """the row's source frame (RGB, image_size square, INTER_AREA) or None when the row has no frame"""
        import av
        import cv2
        tFrame = int(cam["frame_index"][row])
        if tFrame < 0:
            return None
        with av.open(cam["video"]) as tContainer:
            for tK, tDecoded in enumerate(tContainer.decode(video=0)):
                if tK == tFrame:
                    return cv2.resize(tDecoded.to_ndarray(format="rgb24"), (self.mImageSize, self.mImageSize),
                                      interpolation=cv2.INTER_AREA)
