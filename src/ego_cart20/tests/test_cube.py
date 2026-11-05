"""Characterization of the two cube-based metric-scale estimators: cube_scale (pointmap faces) and cube_pnp (silhouette PnP),
on ray-cast pointmaps and rendered fisheye images of a cube of known edge."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego_cart20 import cube_pnp as CP
from ego_cart20 import cube_scale as CS

PNP_K = np.array([[260.0, 0, 160], [0, 260.0, 120], [0, 0, 1]])
PNP_SIZE = (320, 240)
PNP_K_COEFFS = (0.01, -0.005, 0.0, 0.0)
CUBE_BGR = ((0, 40, 225), (10, 60, 245), (5, 30, 200))


# ---------------------------------------------------------------------------------------------------------------- cube_scale
def renderPointmap(edge=0.05, units=1.0, yaw=35, pitch=-30, dist=0.25, occlude=0.0, hw=(192, 256), f=180.0, seed=0):
    """the ray-cast pointmap of tests/cart20/test_cube_scale.py (camera at origin, +z)"""
    tH, tW = hw
    tV, tU = np.mgrid[0:tH, 0:tW]
    tD = np.stack([(tU - tW / 2) / f, (tV - tH / 2) / f, np.ones_like(tU, float)], -1)
    tD /= np.linalg.norm(tD, axis=-1, keepdims=True)
    tR = Rotation.from_euler("yx", [yaw, pitch], degrees=True).as_matrix()
    tC = np.array([0, 0, dist])
    tO = -tC @ tR
    tDl = tD @ tR
    with np.errstate(divide="ignore", invalid="ignore"):
        tT1 = (-edge / 2 - tO) / tDl
        tT2 = (edge / 2 - tO) / tDl
    tNear = np.nanmax(np.minimum(tT1, tT2), -1)
    tFar = np.nanmin(np.maximum(tT1, tT2), -1)
    tHit = (tNear <= tFar) & (tNear > 0)
    tX = np.where(tHit[..., None], tD * tNear[..., None], tD * (dist + 0.3) / tD[..., 2:3])
    tRgb = np.full((tH, tW, 3), 120, np.uint8)
    tRgb[tHit] = (200, 20, 25)
    tX = tX + np.random.default_rng(seed).normal(0, 0.0004, tX.shape)
    if occlude:
        tCut = int(tW * (1 - occlude))
        tX[:, tCut:] = tD[:, tCut:] * 0.08 / tD[:, tCut:, 2:3]
        tRgb[:, tCut:] = (180, 150, 130)
    return (tX * units).astype(np.float32), np.full((tH, tW), 5.0, np.float32), tRgb


def test_cube_scale_primitives(golden):
    golden.check("cs.constants", [CS.CUBE_EDGE_M, CS.MIN_MASK_PX, CS.MIN_FACE_PTS, CS.MAX_RESID_FRAC, CS.SQUARE_TOL,
                                  CS.MAX_VIEW_DEG,
                                  CS.ORTHO_TOL_DEG, CS.MIN_FRAMES, CS.MAX_SPREAD, CS.CONF_MIN])
    golden.check("cs.quat_to_R", CS.quatToR([0.1, -0.2, 0.3, 0.9]))
    tX, tC, tRgb = renderPointmap()
    golden.check("cs.red_mask", CS.redMask(tRgb))
    golden.check("cs.red_mask_eroded", CS.redMask(tRgb, erode=2))
    golden.check("cs.red_mask_empty", CS.redMask(np.full((20, 20, 3), 120, np.uint8)))
    tP = tX[CS.redMask(tRgb)].astype(np.float64)
    tCentre, tNormal, tAxes = CS.fitPlane(tP[:500])
    golden.check("cs.fit_plane", np.concatenate([tCentre, tNormal, tAxes.ravel()]))
    golden.check("cs.ransac_plane", CS.ransacPlane(tP, 0.002))
    golden.check("cs.ransac_plane_degenerate", CS.ransacPlane(np.zeros((5, 3)), 0.01, iters=5) is None)
    golden.check("cs.face_rect", CS.faceRect(np.random.default_rng(0).uniform(-1, 1, (300, 2)) * [2.0, 1.0]))
    tFaces = CS.facesOf(tP, tP / np.linalg.norm(tP, axis=1, keepdims=True))
    golden.check("cs.faces_of", tFaces)


@pytest.mark.parametrize("case", [dict(units=2.7, s_k=0.8), dict(units=1.0, occlude=0.55, s_k=1.0),
                                  dict(units=1.0, s_k=1.0, no_cube=True),
                                  dict(units=1.0, s_k=1.0, dist=2.5), dict(units=1.0, s_k=1.0, yaw=0, pitch=0)])
def test_cube_scale_frame(golden, case):
    tCase = dict(case)
    tSk = tCase.pop("s_k")
    tNoCube = tCase.pop("no_cube", False)
    tX, tC, tRgb = renderPointmap(**tCase)
    if tNoCube:
        tRgb[:] = 120
    golden.check("cs.frame_" + "_".join(f"{k}{v}" for k, v in sorted(case.items())), CS.frameScale(tX, tC, tRgb, tSk, None, 2))


def test_cube_scale_episode(golden, tmp_path):
    tFrames = [renderPointmap(units=2.0, yaw=tY, pitch=tP, seed=tI)
               for tI, (tY, tP) in enumerate([(30, -25), (40, -35), (25, -30), (45, -20)])]
    tRoi = [renderPointmap(units=2.0, yaw=50, pitch=-30, seed=9, hw=(96, 128), f=90.0)]
    np.savez(tmp_path / "kf.npz", X_cam=np.stack([f[0] for f in tFrames]).astype(np.float16), C=np.stack([f[1] for f in tFrames]),
             img=np.stack([f[2] for f in tFrames]), T_WC=np.tile([0, 0, 0, 0, 0, 0, 1, 1.0], (4, 1)), frame_idx=np.arange(4),
             stride=2,
             hw=np.array([384, 512]), K=np.eye(3), roi_src=np.array([10, 11, 12]), roi_X=np.stack([tRoi[0][0]] * 3),
             roi_C=np.stack([tRoi[0][1]] * 3), roi_img=np.stack([tRoi[0][2]] * 3))
    (tmp_path / "ss1.csv").write_text("frame_idx,sim3_scale,is_lost\n10,0.9,False\n11,1.1,True\n")
    tRec = CS.episodeScale(tmp_path / "kf.npz")
    golden.check("cs.episode", tRec)
    golden.check("cs.episode_strip", CS.strip(tRec))
    golden.check("cs.episode_roi_only", CS.episodeScale(tmp_path / "kf.npz", ss1Csv=tmp_path / "ss1.csv", useKeyframes=False))
    golden.check("cs.episode_edge", CS.strip(CS.episodeScale(tmp_path / "kf.npz", edgeM=0.038)))


# ---------------------------------------------------------------------------------------------------------------- cube_pnp
def renderFisheyeCube(tCentre, rCube, edge=CP.CUBE_EDGE_M, size=PNP_SIZE, jaw=False):
    """BGR image of a cube (centre tCentre [m] in the camera frame, rotation rCube) through the KB fisheye PNP_K / PNP_K_COEFFS"""
    import cv2
    tW, tH = size
    tImg = np.full((tH, tW, 3), 120, np.uint8)
    tFaces = []
    for tAxis in range(3):
        for tSign in (1, -1):
            tN = np.zeros(3)
            tN[tAxis] = tSign
            tU, tV = np.roll(np.eye(3), 1, 0)[tAxis], np.roll(np.eye(3), 2, 0)[tAxis]
            tCorners = [(tN + a * tU + b * tV) * edge / 2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            tWorld = np.array([tCentre + rCube @ tX for tX in tCorners])
            tNormal = rCube @ tN
            tFaceC = tCentre + tNormal * edge / 2
            if float(tNormal @ (-tFaceC)) > 0:
                tFaces.append((float(np.linalg.norm(tFaceC)), tWorld))
    for tI, (_, tWorld) in enumerate(sorted(tFaces, key=lambda x: -x[0])):
        tPx, _ = cv2.fisheye.projectPoints(tWorld.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), PNP_K, np.array(PNP_K_COEFFS))
        cv2.fillConvexPoly(tImg, np.round(tPx.reshape(-1, 2) * 16).astype(np.int32), CUBE_BGR[tI % 3], cv2.LINE_AA, 4)
    if jaw:
        tImg[int(tH * 0.62):, :] = 15
    return tImg


def _settingText(path: pathlib.Path) -> pathlib.Path:
    tLines = [f"Camera1.fx: {PNP_K[0, 0]}", f"Camera1.fy: {PNP_K[1, 1]}", f"Camera1.cx: {PNP_K[0, 2]}",
              f"Camera1.cy: {PNP_K[1, 2]}"]
    tLines += [f"Camera1.k{tI + 1}: {tV}" for tI, tV in enumerate(PNP_K_COEFFS)]
    tLines += [f"Camera.width: {PNP_SIZE[0]}", f"Camera.height: {PNP_SIZE[1]}"]
    path.write_text("%YAML:1.0\n" + "\n".join(tLines) + "\n")
    return path


def test_cube_pnp_primitives(golden, tmp_path):
    golden.check("cp.constants", [CP.CUBE_EDGE_M, CP.MIN_AREA_PX, CP.MIN_SOLIDITY, CP.MAX_REPROJ_PX, CP.FACE_MAX_VIEW_DEG,
                                  CP.MIN_FRAMES,
                                  CP.MAX_REL_SPREAD, CP.MAX_CENTRE_RESID_M])
    golden.check("cp.models", {tK: [tV[0], sorted(tV[1])] for tK, tV in CP.MODELS.items()})
    tK, tD, tSize = CP.readSetting(_settingText(tmp_path / "s.yaml"))
    golden.check("cp.read_setting", [tK, tD, list(tSize)])
    tP, tUnd = CP.undistorter(tK, tD, tSize)
    golden.check("cp.undistorter_P", tP)
    golden.check("cp.quat_xyzw_to_R", CP.quatXyzwToR([0.1, -0.2, 0.3, 0.9]))
    tOblique = Rotation.from_euler("yx", [35, -30], degrees=True).as_matrix()
    tTilted = Rotation.from_euler("x", [-35], degrees=True).as_matrix()
    for tName, tCentre, tRot in (("oblique", np.array([0.01, 0.0, 0.22]), tOblique),
                                 ("headon", np.array([0.0, 0.0, 0.2]), np.eye(3)),
                                 ("twoface", np.array([0.0, 0.0, 0.2]), tTilted)):
        tImg = tUnd(renderFisheyeCube(tCentre, tRot))
        tMask = CP.cubeBlob(tImg)
        golden.check(f"cp.blob_{tName}", tMask)
        tHex, tSol = CP.hexagon(tMask)
        golden.check(f"cp.hexagon_{tName}", [tHex, tSol])
        if tHex is not None:
            golden.check(f"cp.pnp_cube_{tName}", CP.pnpCube(tHex, tP))
            if len(tHex) == 4:
                golden.check(f"cp.occluded_{tName}", CP.occludedBelow(tImg, tHex))
                golden.check(f"cp.pnp_face_{tName}", CP.pnpFace(tHex, tP, edge=0.05))
    assert CP.cubeBlob(np.full((240, 320, 3), 120, np.uint8)) is None
    tJaw = tUnd(renderFisheyeCube(np.array([0.0, 0.0, 0.2]), np.eye(3), jaw=True))
    tJawHex, _ = CP.hexagon(CP.cubeBlob(tJaw)) if CP.cubeBlob(tJaw) is not None else (None, None)
    golden.check("cp.jaw_hex", tJawHex)
    if tJawHex is not None and len(tJawHex) == 4:
        golden.check("cp.jaw_occluded", CP.occludedBelow(tJaw, tJawHex))


def test_cube_pnp_sharpen(golden):
    tT = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    tHexTrue = np.array([[np.cos(a), np.sin(a)] for a in np.arange(6) * np.pi / 3]) * 50 + 100
    tContour = []
    for tI in range(6):
        tA, tB = tHexTrue[tI], tHexTrue[(tI + 1) % 6]
        tS = np.linspace(0, 1, 100, endpoint=False)[:, None]
        tContour.append(tA + tS * (tB - tA))
    tContour = np.vstack(tContour) + np.random.default_rng(0).normal(0, 0.2, (600, 2))
    tVerts = tHexTrue * 0.97 + 3
    golden.check("cp.sharpen", CP.sharpen(tContour, tVerts))
    golden.check("cp.sharpen_rev", CP.sharpen(tContour, tVerts[::-1].copy()))
    golden.check("cp.sharpen_ordered", CP.sharpenOrdered(tContour, tVerts, [0, 100, 200, 300, 400, 500], (0.2, 0.8)))
    golden.check("cp.sharpen_dup", CP.sharpen(tContour, np.vstack([tVerts[:5], tVerts[4:5]])))
    assert len(tT) == 600


def test_cube_pnp_fit_scale(golden):
    tRng = np.random.default_rng(1)
    tN = 30
    tC = np.array([0.3, -0.1, 0.5])
    tP = tRng.normal(0, 0.3, (tN, 3))
    tU = (tC - tP) / 2.0 + tRng.normal(0, 0.002, (tN, 3))
    tU[3] += 0.3                                         # one outlier frame
    golden.check("cp.solve_k", CP.solveK(tP, tU))
    golden.check("cp.solve_k_degenerate", CP.solveK(tP, np.zeros_like(tP)))
    golden.check("cp.fit_scale", CP.fitScale(tP, tU))
    golden.check("cp.fit_scale_rng", CP.fitScale(tP, tU, trimIter=1, rng=np.random.default_rng(5)))
    golden.check("cp.fit_scale_neg", CP.fitScale(tP, -tU))


def _pnpEpisode(tmpPath: pathlib.Path, nFrames: int = 16, lostEvery: int = 5):
    import cv2
    tK = 2.0                                                  # world units per metre
    tCubeW = np.array([0.1, 0.05, 0.6])                      # cube centre, metres in the SLAM world
    tWriter = cv2.VideoWriter(str(tmpPath / "v.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, PNP_SIZE)
    tRows = ["frame_idx,x,y,z,q_x,q_y,q_z,q_w,is_lost"]
    tRng = np.random.default_rng(2)
    for tI in range(nFrames):
        tRcw = Rotation.from_euler("yx", [tRng.uniform(-15, 15), tRng.uniform(-10, 10)], degrees=True)
        tCamM = np.array([0.1, 0.05, 0.6]) - tRcw.as_matrix() @ np.array([0.0, 0.0, 0.18 + 0.01 * tI])
        tCentreCam = tRcw.as_matrix().T @ (tCubeW - tCamM)
        tCubeRot = tRcw.as_matrix().T @ Rotation.from_euler("yx", [35, -30], degrees=True).as_matrix()
        tWriter.write(renderFisheyeCube(tCentreCam, tCubeRot))
        tQ = tRcw.as_quat()
        tPw = tCamM * tK
        tLost = "True" if tI % lostEvery == 4 else "False"
        tRows.append(f"{tI},{tPw[0]},{tPw[1]},{tPw[2]},{tQ[0]},{tQ[1]},{tQ[2]},{tQ[3]},{tLost}")
    tWriter.release()
    (tmpPath / "ss1.csv").write_text("\n".join(tRows) + "\n")
    return tmpPath / "v.avi", _settingText(tmpPath / "s.yaml"), tmpPath / "ss1.csv"


def _strip(rec):
    return {tK: tV for tK, tV in rec.items() if tK != "frames"}


def test_episode_pnp_scale(golden, tmp_path):
    tVideo, tSetting, tCsv = _pnpEpisode(tmp_path)
    tRec = CP.episodePnpScale(tVideo, tSetting, tCsv)
    golden.check("cp.episode_frames_ok", [tF["ok"] for tF in tRec["frames"]])
    golden.check("cp.episode", _strip(tRec))
    golden.check("cp.episode_frac", _strip(CP.episodePnpScale(tVideo, tSetting, tCsv, stride=2, maxFrac=0.5)))
    golden.check("cp.episode_edge", _strip(CP.episodePnpScale(tVideo, tSetting, tCsv, edge=0.05)))
