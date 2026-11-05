"""Known-size metric scale from the red cube, in the SAME units as the MASt3R-SLAM trajectory (HRA_red, 2026-10-03).

Why: on the 10 s right-only approach takes the IMU-VI scale is unobservable (pilot s = 0.084 / 0.020 vs ego p50 0.36-0.50;
displacements came out 3-8 cm).  User decision: per-episode known-size scale, never "assume the approach was N cm".

Input per side: <runs>/<tag>/ss1.kf.npz from run_perframe_resident_kfdump.py -- per keyframe the pointmap X in the
keyframe's camera frame, its confidence C, the RGB MASt3R saw, and the FINAL keyframe Sim3 T_WC = [t, q xyzw, s].
World (= trajectory) lengths are s_k * camera-frame lengths, so every length below is multiplied by s_k.

Per keyframe:
    red mask (HSV, largest blob; NOT eroded: erosion shrank faces 5-17 % in the synthetic test -- flying pixels at the
    silhouette are off every face plane and fall out of the RANSAC inliers instead)
    -> 3D points (confidence-gated) -> sequential RANSAC planes (<= 3 faces), each must be ~orthogonal to the others
    -> per face: minimum-area rectangle of the in-plane points (1 % farthest trimmed) = two edge estimates
       a face counts only if it is square-ish (occlusion by the hand shrinks one side -> rectangle -> rejected),
       is not seen edge-on (angle between normal and view ray < MAX_VIEW_DEG), has enough points and a small residual
    -> L_raw(frame) = median of the accepted face edges (world units);  s_frame = L_real / L_raw
Per episode:  s_cube = median(s_frame) over >= MIN_FRAMES accepted frames with spread (MAD / median) <= MAX_SPREAD,
              else INVALID with the reason.  Every number goes into the QC record (user list 2026-10-03).

Pipeline stage: pose stage, metric scale (alternative to the IMU-VI scale and to cube_pnp.py).
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np

# numpy 2 + macOS Accelerate: spurious matmul warnings (installed on import, as before the restyle)
warnings.filterwarnings("ignore", message=".*encountered in matmul")
CUBE_EDGE_M = 0.05             # the task cube.  CONFIRM with the operator before freezing the dataset
MIN_MASK_PX = 150              # at the dumped (stride-2) resolution
MIN_FACE_PTS = 60
MAX_RESID_FRAC = 0.06          # plane RMS residual / edge estimate
SQUARE_TOL = 0.25              # face aspect must be within 1 +- 0.25
MAX_VIEW_DEG = 70.0            # face normal vs view ray
ORTHO_TOL_DEG = 20.0           # extra faces must be 90 +- 20 deg from the first
MIN_FRAMES = 3
MAX_SPREAD = 0.15              # MAD / median of s_frame inside an episode
CONF_MIN = 1.5                 # MASt3R confidence (same default C_conf_threshold as the SLAM viewer)

# red cube in OpenCV HSV (hue 0..180): the HRA cube is orange-red, hue ~10, sat ~230
RED_HUE_LOW_MAX = 18
RED_HUE_HIGH_MIN = 170
RED_SAT_MIN = 150
RED_VAL_MIN = 50
OPEN_KERNEL_PX = 3
CLOSE_KERNEL_PX = 5
RANSAC_ITERS = 200
RANSAC_THR_FRAC = 0.04         # plane inlier distance = this fraction of the blob's 3-97 % extent diagonal
EXTENT_PCT = (3, 97)
NORM_EPS = 1e-9               # keeps a threshold / divisor away from zero
DEGENERATE_NORMAL = 1e-12      # a 3-point sample with a shorter normal is collinear
FACE_TRIM = 0.01
MAX_FACES = 3
TINY = 1e-12


def quatToR(q) -> np.ndarray:
    """xyzw quaternion -> rotation matrix (was quat_to_R)"""
    tX, tY, tZ, tW = q
    return np.array([[1 - 2 * (tY * tY + tZ * tZ), 2 * (tX * tY - tZ * tW), 2 * (tX * tZ + tY * tW)],
                     [2 * (tX * tY + tZ * tW), 1 - 2 * (tX * tX + tZ * tZ), 2 * (tY * tZ - tX * tW)],
                     [2 * (tX * tZ - tY * tW), 2 * (tY * tZ + tX * tW), 1 - 2 * (tX * tX + tY * tY)]])


def redMask(rgb: np.ndarray, *, erode: int = 0) -> np.ndarray:
    """largest orange-red blob of an RGB image (bool mask, all False if none)"""
    import cv2
    tHsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    tH, tS, tV = tHsv[..., 0].astype(int), tHsv[..., 1].astype(int), tHsv[..., 2].astype(int)
    tMask = (((tH <= RED_HUE_LOW_MAX) | (tH >= RED_HUE_HIGH_MIN)) & (tS >= RED_SAT_MIN) & (tV >= RED_VAL_MIN)).astype(np.uint8)
    tMask = cv2.morphologyEx(tMask, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL_PX, OPEN_KERNEL_PX), np.uint8))
    tMask = cv2.morphologyEx(tMask, cv2.MORPH_CLOSE, np.ones((CLOSE_KERNEL_PX, CLOSE_KERNEL_PX), np.uint8))
    tN, tLabels, tStats, _ = cv2.connectedComponentsWithStats(tMask)
    if tN <= 1:
        return np.zeros_like(tMask, bool)
    tK = 1 + int(np.argmax(tStats[1:, cv2.CC_STAT_AREA]))
    tMask = (tLabels == tK).astype(np.uint8)
    if erode:
        tMask = cv2.erode(tMask, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    return tMask.astype(bool)


def fitPlane(points):
    """least-squares plane -> (centre, normal, two in-plane axes (2, 3))"""
    tC = points.mean(0)
    _, _, tVt = np.linalg.svd(points - tC, full_matrices=False)
    return tC, tVt[2], tVt[:2]


def ransacPlane(points, thr, iters=RANSAC_ITERS, rng=None):
    """inlier mask of the best 3-point plane (None if every sample was degenerate)"""
    tRng = rng or np.random.default_rng(0)
    tBest = None
    for _ in range(iters):
        tI = tRng.choice(len(points), 3, replace=False)
        tN = np.cross(points[tI[1]] - points[tI[0]], points[tI[2]] - points[tI[0]])
        if np.linalg.norm(tN) < DEGENERATE_NORMAL:
            continue
        tN /= np.linalg.norm(tN)
        tInliers = np.abs((points - points[tI[0]]) @ tN) < thr
        if tBest is None or tInliers.sum() > tBest.sum():
            tBest = tInliers
    return tBest


def faceRect(uv, trim=FACE_TRIM) -> np.ndarray:
    """side lengths of the minimum-area rectangle around the in-plane points, after dropping the `trim` fraction farthest
    from the median.  Orientation-free: a PCA extent is undefined for a square (isotropic) face and reads up to sqrt(2) L."""
    import cv2
    tD = np.linalg.norm(uv - np.median(uv, 0), axis=1)
    tKeep = tD <= np.quantile(tD, 1 - trim)
    (_, _), (tW, tH), _ = cv2.minAreaRect(uv[tKeep].astype(np.float32))
    return np.array([tW, tH], float)


def facesOf(points, viewDirs, *, maxFaces=MAX_FACES) -> list:
    """sequential RANSAC faces. points (n,3) camera-frame points (any unit), viewDirs (n,3) unit rays."""
    tOut, tRest = [], np.arange(len(points))
    tExtent = np.percentile(points, EXTENT_PCT[1], axis=0) - np.percentile(points, EXTENT_PCT[0], axis=0)
    tThr = RANSAC_THR_FRAC * float(np.linalg.norm(tExtent)) + NORM_EPS
    for _ in range(maxFaces):
        if len(tRest) < MIN_FACE_PTS:
            break
        tInliers = ransacPlane(points[tRest], tThr)
        if tInliers is None or tInliers.sum() < MIN_FACE_PTS:
            break
        tIdx = tRest[tInliers]
        tC, tN, tAxes = fitPlane(points[tIdx])
        tResid = float(np.sqrt(np.mean(((points[tIdx] - tC) @ tN) ** 2)))
        tEdges = faceRect(((points[tIdx] - tC) @ tAxes.T))
        tMeanRay = np.mean(viewDirs[tIdx], 0)
        tView = float(np.degrees(np.arccos(min(1.0, abs(float(tMeanRay @ tN) / max(np.linalg.norm(tMeanRay), NORM_EPS))))))
        tOut.append(dict(n_pts=int(len(tIdx)), normal=tN, edges=tEdges, resid=tResid, view_deg=tView))
        tRest = tRest[~tInliers]
    return tOut


def frameScale(xCam, conf, rgb, sK, k, stride, *, edgeM=CUBE_EDGE_M) -> dict:
    """one keyframe: pointmap xCam (H, W, 3), confidence (H, W), RGB, keyframe Sim3 scale sK -> QC record with s_frame
    (k and stride are unused; kept in the signature of the kfdump call)"""
    tMask = redMask(rgb)
    tRec = dict(mask_px=int(tMask.sum()))
    if tMask.sum() < MIN_MASK_PX:
        return dict(tRec, ok=False, why="mask_small")
    tGood = tMask & (conf >= CONF_MIN) & np.isfinite(xCam).all(-1)
    tP = xCam[tGood].astype(np.float64)
    tRec["pts"] = int(len(tP))
    if len(tP) < MIN_FACE_PTS:
        return dict(tRec, ok=False, why="few_points")
    tViewDirs = tP / np.linalg.norm(tP, axis=1, keepdims=True)
    tFaces = facesOf(tP, tViewDirs)
    tAccepted = []
    tWhy = []
    for tJ, tFace in enumerate(tFaces):
        if tJ and abs(float(tFace["normal"] @ tFaces[0]["normal"])) > np.sin(np.radians(ORTHO_TOL_DEG)):
            tWhy.append("not_orthogonal")
            continue
        tA, tB = sorted(tFace["edges"])
        tL = float(np.sqrt(tA * tB))
        if tB / max(tA, TINY) > 1 + SQUARE_TOL:
            tWhy.append("occluded_or_partial")
            continue
        if tFace["view_deg"] > MAX_VIEW_DEG:
            tWhy.append("edge_on")
            continue
        if tFace["resid"] / max(tL, TINY) > MAX_RESID_FRAC:
            tWhy.append("residual")
            continue
        tAccepted.append(dict(L_cam=tL, a=tA, b=tB, resid_frac=tFace["resid"] / tL, view_deg=tFace["view_deg"],
                              n_pts=tFace["n_pts"]))
    tRec.update(faces=len(tFaces), faces_ok=len(tAccepted), reject=tWhy)
    if not tAccepted:
        return dict(tRec, ok=False, why="no_valid_face:" + ",".join(sorted(set(tWhy))) if tWhy else "no_face")
    tLRaw = float(sK * np.median([tF["L_cam"] for tF in tAccepted]))
    return dict(tRec, ok=True, L_raw=tLRaw, s_frame=edgeM / tLRaw,
                face_resid_frac=float(np.median([tF["resid_frac"] for tF in tAccepted])),
                aspect=float(np.median([tF["b"] / tF["a"] for tF in tAccepted])),
                view_deg=float(np.median([tF["view_deg"] for tF in tAccepted])))


def episodeScale(kfNpz: pathlib.Path, *, ss1Csv: pathlib.Path | None = None, edgeM=CUBE_EDGE_M,
                 useKeyframes: bool = True) -> dict:
    """keyframes (final Sim3 scale) + tracked-frame red ROIs at full resolution (Sim3 scale of that frame from ss1.csv,
    the same re-expression on the final keyframes the trajectory uses)"""
    tZ = np.load(kfNpz, allow_pickle=True)
    tFrames = []
    if useKeyframes and len(tZ["T_WC"]):
        tX, tC, tImg, tTwc = tZ["X_cam"].astype(np.float32), tZ["C"].astype(np.float32), tZ["img"], tZ["T_WC"]
        for tK in range(len(tTwc)):
            tRec = frameScale(tX[tK], tC[tK], tImg[tK], float(tTwc[tK, 7]), tZ["K"], int(tZ["stride"]), edgeM=edgeM)
            tRec.update(src="kf", kf=tK, frame_idx=int(tZ["frame_idx"][tK]))
            tFrames.append(tRec)
    if "roi_src" in tZ.files and len(tZ["roi_src"]):
        import pandas as pd
        tSs = pd.read_csv(ss1Csv if ss1Csv else pathlib.Path(kfNpz).with_name("ss1.csv")).set_index("frame_idx")
        for tJ, tSrc in enumerate(tZ["roi_src"]):
            tSrc = int(tSrc)
            if tSrc not in tSs.index or str(tSs.loc[tSrc, "is_lost"]).lower() == "true":
                continue
            tRec = frameScale(tZ["roi_X"][tJ].astype(np.float32), tZ["roi_C"][tJ].astype(np.float32), tZ["roi_img"][tJ],
                              float(tSs.loc[tSrc, "sim3_scale"]), tZ["K"], 1, edgeM=edgeM)
            tRec.update(src="roi", frame_idx=tSrc)
            tFrames.append(tRec)
    tOk = [tF for tF in tFrames if tF["ok"]]
    tS = np.array([tF["s_frame"] for tF in tOk])
    tOut = dict(n_keyframes=len(tFrames), n_frames_ok=len(tOk), cube_edge_m=edgeM, frames=tFrames,
                reject_counts={tW: sum(1 for tF in tFrames if not tF["ok"] and tF["why"].startswith(tW.split(":")[0]))
                               for tW in sorted(set(tF["why"].split(":")[0] for tF in tFrames if not tF["ok"]))})
    if len(tOk) < MIN_FRAMES:
        return dict(tOut, valid=False, reason=f"only {len(tOk)} keyframes with a measurable cube (< {MIN_FRAMES})")
    tMed = float(np.median(tS))
    tSpread = float(np.median(np.abs(tS - tMed)) / tMed)
    tOut.update(s_cube=tMed, spread=tSpread, cube_edge_raw=float(edgeM / tMed),
                fit_residual=float(np.median([tF["face_resid_frac"] for tF in tOk])),
                num_cube_points=int(np.median([tF["pts"] for tF in tOk])))
    if tSpread > MAX_SPREAD:
        return dict(tOut, valid=False, reason=f"s_frame spread {tSpread:.2f} > {MAX_SPREAD}")
    return dict(tOut, valid=True, reason="")


def strip(rec: dict) -> dict:
    """QC row without the per-frame detail"""
    return {tK: tV for tK, tV in rec.items() if tK != "frames"}
