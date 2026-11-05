"""CART-ONLY pre-training preflight (CPU, no GPU) for ego_relcart20task_rel16_v4_nopseudoq-style datasets (training stage, node).

[2026-09-30] Replaces preflight_rel16v2.py for the ego cart-only pretrain (that one requires aux.q_t at action dim 32). Same code
path as lerobot_train (base config + launcher overrides -> resolve_delta_timestamps -> LeRobotDataset -> make_policy /
make_pre_post_processors with the dataset stats) with rel16_aux_masked installed. Asserts:
  P1 no delta expansion on action; sample action (16, 32), state (sdim,), bit-identical to the stored parquet row
  P2 dataset has no aux.q_t / aux.fk_mask feature; has_dq_supervision == has_fk_supervision == 0 on every row (parquet scan)
  P3 action dims 0:20 finite, dims 20:32 NaN on every row
  P4 full policy forward + backward on one batch: total loss finite, rel_loss finite > 0, dq_loss == fk_loss == 0 exactly
  P5 no q_t access: batch carries no aux.q_t, action space _rel16_qt is None, FK (ReBotFKTorch.tcp) called 0 times
  P6 action_decoder grad for output dims 20:32 exactly 0 in every domain, dims 0:20 non-zero
Output: PASS / FAIL lines, "PREFLIGHT CARTONLY PASS" and exit 0, or exit 1 on any failure.
usage: preflight_cartonly.py <dataset root> <base dir> <state dim>     (env REL16_STATS = <dataset root>/meta/stats.json)
       (node launcher scripts/train/node4090/train_ego_cartonly_v4.sh; needs LeRobot)
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ego_paths import NODE_HOME, prepend_sys_path  # machine-specific roots (env-overridable)

try:
    from . import rel16_aux_masked as RM
except ImportError:  # run as a flat script on the node (its folder is sys.path[0])
    import rel16_aux_masked as RM

RENAME = {"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2",
          "observation.images.right_wrist": "observation.images.image3"}
CHUNK = 16
ACTION_DIMS = 32
REL_DIMS = 20


class Preflight:
    """PASS / FAIL printer that remembers the failed messages"""

    def __init__(self):
        self.mFails = []

    def need(self, ok, msg):
        print(("PASS  " if ok else "FAIL  ") + msg, flush=True)
        if not ok:
            self.mFails.append(msg)


def _column(t, c):
    tArr = t.column(c).combine_chunks()
    return tArr.storage if isinstance(tArr, pa.ExtensionArray) else tArr


def _loadFkModule():
    prepend_sys_path(str(NODE_HOME / "c8old/xvla"))
    try:
        import rebot_fk_torch  # the node / Mac copy on the path above (unchanged behaviour)
    except ImportError:  # clean checkout: the repository copy, src/training/robot/rebot_fk_torch.py
        from training.robot import rebot_fk_torch
    return rebot_fk_torch


def _countFkCalls(fkModule):
    """wrap ReBotFKTorch.tcp with a call counter; -> the counter list"""
    tCalls = [0]
    tTcp = fkModule.ReBotFKTorch.tcp

    def tcp(self, *a, **k):
        tCalls[0] += 1
        return tTcp(self, *a, **k)
    fkModule.ReBotFKTorch.tcp = tcp
    return tCalls


def _scanParquet(root):
    """-> (actions of the first file, all actions, all mask columns, `index` of the first file)"""
    tA, tM = [], []
    tIndex = None
    for tFile in sorted(glob.glob(f"{root}/data/chunk-*/*.parquet")):
        tTable = pq.read_table(tFile, columns=["action", "index"] + list(RM.MASK_KEYS))
        tA.append(np.asarray(_column(tTable, "action").to_pylist(), np.float32))
        tM.append(np.stack([_column(tTable, tK).to_numpy() for tK in RM.MASK_KEYS], 1))
        tIndex = _column(tTable, "index").to_numpy() if len(tA) == 1 else tIndex
    return tA[0], np.concatenate(tA), np.concatenate(tM), tIndex


def run(root, base, sdim) -> bool:
    """all checks; -> True when every check passed"""
    import lerobot.policies.factory as factory
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    tPf = Preflight()
    tFkCalls = _countFkCalls(_loadFkModule())

    tCfg = PreTrainedConfig.from_pretrained(base)
    tCfg.pretrained_path = base
    for tK, tV in dict(device="cpu", dtype="float32", chunk_size=CHUNK, n_action_steps=CHUNK, max_state_dim=sdim,
                       max_action_dim=ACTION_DIMS, action_mode="auto", use_proprio=True, freeze_vision_encoder=False,
                       freeze_language_encoder=False, train_policy_transformer=True, train_soft_prompts=True).items():
        setattr(tCfg, tK, tV)
    tMeta = LeRobotDatasetMetadata("rebot/preflight", root=root)
    tDts = resolve_delta_timestamps(tCfg, tMeta)
    tPf.need("action" not in (tDts or {}),
             f"P1 resolve_delta_timestamps leaves action unexpanded (delta keys {sorted((tDts or {}).keys())})")
    tPf.need(not any(tK in tMeta.features for tK in ("aux.q_t", "aux.fk_mask")) and all(tK in tMeta.features for tK in RM.MASK_KEYS),
             "P2 features: no aux.q_t / aux.fk_mask, mask columns present")

    tA0, tA, tM, tIx = _scanParquet(root)
    tPf.need(bool((tM == 0).all()), f"P2 has_dq / has_fk == 0 on all {len(tM)} rows")
    tPf.need(bool(np.isfinite(tA[..., :REL_DIMS]).all() and np.isnan(tA[..., REL_DIMS:]).all()),
             "P3 action dims 0:20 finite, dims 20:32 NaN on every row")

    tDs = LeRobotDataset("rebot/preflight", root=root, delta_timestamps=tDts, video_backend="pyav")
    tItems = [tDs[tI] for tI in (0, len(tDs) // 2)]
    for tX in tItems:
        tJ = int(tX["index"])
        tStored = tA0[np.flatnonzero(tIx == tJ)[0]] if tJ in tIx else None
        tPf.need(tuple(tX["action"].shape) == (CHUNK, ACTION_DIMS) and tuple(tX["observation.state"].shape) == (sdim,),
                 f"P1 sample {tJ}: action {tuple(tX['action'].shape)} state {tuple(tX['observation.state'].shape)}")
        if tStored is not None:
            tPf.need(np.array_equal(tX["action"].numpy(), tStored, equal_nan=True),
                     f"P1 sample {tJ}: loader action == stored parquet row (bit-for-bit, NaN-aware)")

    tPolicy = factory.make_policy(cfg=tCfg, ds_meta=tMeta, rename_map=RENAME)
    tPolicy.train()
    tPre, _ = factory.make_pre_post_processors(policy_cfg=tCfg, pretrained_path=base, preprocessor_overrides={
        "device_processor": {"device": "cpu"}, "rename_observations_processor": {"rename_map": RENAME},
        "normalizer_processor": {"stats": tMeta.stats, "features": {**tPolicy.config.input_features, **tPolicy.config.output_features},
                                 "norm_map": tPolicy.config.normalization_mapping}})
    tBatch = tPre(torch.utils.data.default_collate(tItems))
    tPf.need("aux.q_t" not in tBatch and all(tK in tBatch for tK in RM.MASK_KEYS), "P5 batch has mask keys and no aux.q_t")
    tLoss, tLog = tPolicy.forward(tBatch)
    tLoss.backward()
    tPf.need(bool(torch.isfinite(tLoss)) and tLog["rel_loss"] > 0, f"P4 losses finite: {tLog}")
    tPf.need(tLog["dq_loss"] == 0.0 and tLog["fk_loss"] == 0.0, "P4 dq_loss == fk_loss == 0 exactly")
    tPf.need(tPolicy.model.action_space._rel16_qt is None and tFkCalls[0] == 0,
             f"P5 no q_t access (action space q_t None, FK calls {tFkCalls[0]})")
    tDec = [tMod for tN, tMod in tPolicy.named_modules() if tN.endswith("action_decoder")][0]
    tW = tDec.fc.weight.grad.view(-1, tDec.input_size, tDec.output_size)
    tBg = tDec.bias.weight.grad
    tPf.need(bool((tW[..., REL_DIMS:ACTION_DIMS] == 0).all() and (tBg[:, REL_DIMS:ACTION_DIMS] == 0).all()
                  and (tW[..., :REL_DIMS].abs() > 0).any()),
             "P6 action_decoder grad: dims 20:32 exactly 0 (all domains), dims 0:20 non-zero")
    print("PREFLIGHT CARTONLY PASS" if not tPf.mFails else f"PREFLIGHT CARTONLY FAILED: {tPf.mFails}")
    return not tPf.mFails


def main(argv: list[str] | None = None) -> int:
    tArgv = sys.argv[1:] if argv is None else argv
    tRoot, tBase, tSdim = tArgv[0], tArgv[1], int(tArgv[2])
    RM.install()
    return 0 if run(tRoot, tBase, tSdim) else 1


if __name__ == "__main__":
    sys.exit(main())
