"""[2026-09-25] imu_vi_g: cp6_scale.imu_vi with the SAME source (exec of the frozen text), whose return additionally carries
the gravity vector g_world = xs[1:4] (Gen-1 census stage; was c8_imu_vi_g.py).

In:  ~/umi_bridge/trackA_mast3r_pose_v1/cp6_scale.py (the frozen C-P6 text, exec'd up to its FIT/EVAL section), and the
     staging tree it reads through M3_BASE / M3_VARIANT (like phase3_segments' stage_p3)
Out: loadFrozenFit() -> namespace of the frozen text with imu_vi (with g_world) and imu_vi_ref (the unchanged one);
     run as a script: per tag IDENTICAL / DIFFERENT and the final IMU_VI_G_VERIFIED / IMU_VI_G_MISMATCH <n>
Usage: python -m rebot_prep.census.imu_vi_gravity <tag> [<tag> ...]     (M3_BASE defaults to ~/c8/stage_p3)

g_world is in that side's MASt3R world; model R_wb f_b - a_lever = s a_cam - g, so at rest g = -R_wb f_b = physical
gravity, pointing DOWN. The only textual change is the return line; s / g_norm / n / resid are verified bit-identical to
cp6's imu_vi before use. The namespace keys (imu_vi, imu_vi_ref, load_traj, X, ...) are the frozen text's names.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from ego_paths import DATA_ROOT, UMI_BRIDGE

TA = UMI_BRIDGE / "trackA_mast3r_pose_v1"
RET = 'return dict(s=float(xs[0]), g_norm=float(np.linalg.norm(xs[1:4])), n=int(len(y) // 3),'
FROZEN_END = "tags = ["                   # the frozen text is exec'd up to its FIT/EVAL section
VERIFY_WINDOW, VERIFY_BIAS = 15, False    # the frozen C-P6 choice


class FrozenSourceError(RuntimeError):
    """The frozen cp6_scale.py text no longer has exactly one imu_vi return line: re-verify imu_vi_g."""


def loadFrozenFit() -> dict:
    """exec the frozen text twice: unchanged (imu_vi_ref) and with g_world in the return (imu_vi) (was load)"""
    tSrc = open(TA / "cp6_scale.py").read()
    tPre = tSrc[:tSrc.index(FROZEN_END)]
    if tPre.count(RET) != 1:
        raise FrozenSourceError("cp6_scale.imu_vi return line changed -- re-verify imu_vi_g")
    tRef, tWithG = {}, {}
    exec(tPre, tRef)
    exec(tPre.replace(RET, RET.replace("dict(", "dict(g_world=xs[1:4].tolist(), ")), tWithG)
    tWithG["imu_vi_ref"] = tRef["imu_vi"]
    return tWithG


def main(argv: list[str] | None = None) -> int:
    tTags = sys.argv[1:] if argv is None else argv
    os.environ.setdefault("M3_BASE", str(DATA_ROOT / "stage_p3"))
    os.environ.setdefault("M3_VARIANT", "ss1")
    tNs = loadFrozenFit()
    tBad = 0
    for tTag in tTags:
        tA, tB = tNs["imu_vi_ref"](tTag, VERIFY_WINDOW, VERIFY_BIAS), tNs["imu_vi"](tTag, VERIFY_WINDOW, VERIFY_BIAS)
        tSame = (tA is None and tB is None) or all(tA[tK] == tB[tK] for tK in tA)
        tG = np.array(tB["g_world"]) if tB else None
        print(tTag, "IDENTICAL" if tSame else "DIFFERENT",
              f"s {tB['s']:.6f} |g| {tB['g_norm']:.4f} ‖g_world‖ {np.linalg.norm(tG):.4f}" if tB else "none")
        tBad += not tSame
    print("IMU_VI_G_VERIFIED" if tBad == 0 else f"IMU_VI_G_MISMATCH {tBad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
