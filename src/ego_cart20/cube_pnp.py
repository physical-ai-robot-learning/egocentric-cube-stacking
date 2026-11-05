"""Metric scale of a MASt3R-SLAM trajectory from a STATIC cube of known edge, by PnP (HRA_red, 2026-10-03, user design).

The cube is the calibration object.  Per frame of the SLAM input video (the exact frames ss1.csv indexes):
    KB-fisheye -> pinhole (cv2.fisheye, balance 0 = the camera MASt3R saw: same axes, R = I)
    orange-red blob -> convex hull -> hexagon (3 faces visible) -> 12 cyclic assignments to the cube's silhouette corners
    -> solvePnP, best reprojection with cheirality (the hexagon's "near" corner must be the corner closest to the camera)
    -> t_i = cube centre in the camera frame, METRES.  t_i is invariant to the cube's 48 symmetries, so a corner
       labelling that flips between frames does not matter.
SLAM gives camera->world (R_i, p_i) in arbitrary world units.  The cube does not move, so with k = world units per metre
    c_i = p_i + k * R_i @ t_i      must be the same point for every frame
    k = - sum (p_i - p_mean) . (u_i - u_mean) / sum |u_i - u_mean|^2,   u_i = R_i t_i      (closed form, then trimmed)
    metric scale s = 1 / k  (metres per world unit; the trajectory times s is in metres)
Nothing here uses MASt3R depth, the pointmap, the Sim3 scale column, or the IMU.  QC per episode: valid PnP frames,
reprojection error, scale spread (bootstrap), cube-centre residual in metres.

In: SLAM input video + ORB-SLAM-style camera setting + ss1.csv.  Out: QC dict with s_pnp.
Pipeline stage: pose stage, metric scale (alternative to the IMU-VI scale).
"""
from __future__ import annotations

import pathlib
import re

import numpy as np

CUBE_EDGE_M = 0.038            # measured by the operator 2026-10-03 (the HRL80 / HRA stacking cube set). 0.05 was a guess
MIN_AREA_PX = 400              # blob area at 960x540 (pinhole)
MIN_SOLIDITY = 0.88
MAX_REPROJ_PX = 3.0
FACE_MAX_VIEW_DEG = 35.0      # single-face model: face normal vs line of sight
# validity is judged by the bootstrap spread + centre residual gates; 10 frames already over-determine one scalar
MIN_FRAMES = 10
MAX_REL_SPREAD = 0.10          # bootstrap (p84 - p16) / 2 / median
MAX_CENTRE_RESID_M = 0.02

# orange-red blob in OpenCV HSV (hue 0..180)
RED_HUE_LOW_MAX = 18
RED_HUE_HIGH_MIN = 170
RED_SAT_MIN = 150
RED_VAL_MIN = 50
OPEN_KERNEL_PX = 3
CLOSE_KERNEL_PX = 5
BORDER_PX = 1                  # a blob touching this margin is cut by the image border
HULL_EPS_FRACS = np.linspace(0.01, 0.08, 36)   # approxPolyDP tolerances tried, as fractions of the hull perimeter
SHARPEN_KEEP = (0.2, 0.8)      # middle part of each silhouette side used for the line fit
PARALLEL_DET = 1e-6            # two side lines closer to parallel than this keep the polygon vertex
SHARPEN_MAX_SHIFT_FRAC = 0.25  # a sharpened corner may move at most this fraction of its neighbours' distance
OCCLUSION_BAND_PX = 8
OCCLUSION_DARK = 90            # median grey below this = the black gripper jaw
OCCLUSION_SAMPLES = 20
OCCLUSION_OFFSET_PX = 3
VISIBLE_FACE_MARGIN = 1e-3     # camera must be this fraction beyond a face plane to see that face
TRIM_ITERS = 3
TRIM_MAD_K = 4                 # keep residuals within median + 4 sigma
MAD_TO_SIGMA = 1.4826
MAD_EPS = 1e-9
K_EPS = 1e-12
BOOTSTRAP_N = 200
BOOTSTRAP_PCT = (16, 84)
CONTOUR_EPS = 1e-9
CM_PER_M = 100

# cube silhouette from the octant of the near corner (+,+,+): alternating single / double sign flips, cyclic
_SIL = np.array([(-1, 1, 1), (-1, -1, 1), (1, -1, 1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)], float)
# two visible faces (+z top, +y toward the camera): the silhouette runs over the top-back edge, down the front-right edge,
# along the front-bottom edge and up the front-left edge (measured: the far approach frames see exactly top + front)
_SIL2 = np.array([(-1, -1, 1), (1, -1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1), (-1, 1, 1)], float)
MODELS = {"3face": (_SIL, {(1, 0, 0), (0, 1, 0), (0, 0, 1)}), "2face": (_SIL2, {(0, 1, 0), (0, 0, 1)})}
_FACES = [tuple(int(v) for v in n) for n in np.vstack([np.eye(3), -np.eye(3)])]
_CORNERS = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)


def readSetting(path):
    """ORB-SLAM-style camera yaml -> (K 3x3, KB coefficients (4, 1), (width, height))"""
    tText = pathlib.Path(path).read_text()

    def _value(key):
        return float(re.search(rf"^{key}:\s*([-\d.eE]+)", tText, re.M).group(1))

    tK = np.array([[_value("Camera1.fx"), 0, _value("Camera1.cx")], [0, _value("Camera1.fy"), _value("Camera1.cy")], [0, 0, 1.0]])
    tD = np.array([_value(f"Camera1.k{tI}") for tI in (1, 2, 3, 4)]).reshape(4, 1)
    return tK, tD, (int(_value("Camera.width")), int(_value("Camera.height")))


def undistorter(k, d, size):
    """-> (pinhole P, function image -> undistorted image) for balance 0 (same axes as the fisheye camera)"""
    import cv2
    tP = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, size, np.eye(3), balance=0.0)
    tMapX, tMapY = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), tP, size, cv2.CV_32FC1)
    return tP, (lambda img: cv2.remap(img, tMapX, tMapY, cv2.INTER_LINEAR))


def cubeBlob(bgr):
    """uint8 mask of the largest orange-red blob, or None if there is none / it is too small / it touches the border"""
    import cv2
    tHsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    tH, tS, tV = tHsv[..., 0], tHsv[..., 1], tHsv[..., 2]
    tMask = (((tH <= RED_HUE_LOW_MAX) | (tH >= RED_HUE_HIGH_MIN)) & (tS >= RED_SAT_MIN) & (tV >= RED_VAL_MIN)).astype(np.uint8)
    tMask = cv2.morphologyEx(tMask, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL_PX, OPEN_KERNEL_PX), np.uint8))
    tMask = cv2.morphologyEx(tMask, cv2.MORPH_CLOSE, np.ones((CLOSE_KERNEL_PX, CLOSE_KERNEL_PX), np.uint8))
    tN, tLabels, tStats, _ = cv2.connectedComponentsWithStats(tMask)
    if tN <= 1:
        return None
    tK = 1 + int(np.argmax(tStats[1:, cv2.CC_STAT_AREA]))
    tX, tY, tW, tHh, tArea = tStats[tK]
    tHeight, tWidth = tMask.shape
    if (tArea < MIN_AREA_PX or tX <= BORDER_PX or tY <= BORDER_PX or tX + tW >= tWidth - BORDER_PX
            or tY + tHh >= tHeight - BORDER_PX):
        return None     # cut by the border
    return (tLabels == tK).astype(np.uint8)


def hexagon(mask):
    """-> (silhouette corners (6, 2) sharpened, or (4, 2) for a one-face view, or None; solidity)"""
    import cv2
    tContours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tContour = max(tContours, key=cv2.contourArea)
    tHull = cv2.convexHull(tContour)
    tSolidity = cv2.contourArea(tContour) / max(cv2.contourArea(tHull), CONTOUR_EPS)
    tPerimeter = cv2.arcLength(tHull, True)
    tQuad = None
    for tEps in HULL_EPS_FRACS:
        tApprox = cv2.approxPolyDP(tHull, tEps * tPerimeter, True)
        if len(tApprox) == 6:
            return sharpen(tContour.reshape(-1, 2).astype(np.float64), tApprox.reshape(6, 2).astype(np.float64)), tSolidity
        if len(tApprox) == 4 and tQuad is None:
            tQuad = tApprox.reshape(4, 2).astype(np.float64)
        if len(tApprox) < 4:
            break
    # [2026-10-03] a top-down approach sees ONE face: a quadrilateral silhouette (measured on HRA ep 150 / 180, where the
    # hexagon models found 0-2 frames).  Corners from the hull vertices (no 6-side sharpening for 4 sides).
    if tQuad is not None:
        return tQuad, tSolidity
    return None, tSolidity


def sharpen(contour, verts, keep=SHARPEN_KEEP):
    """corner = intersection of the lines fitted to the two adjacent silhouette sides (middle 60 % of each side's contour
    points).  A polygon vertex taken ON the contour sits inside the true corner: the physical cube edges are rounded and
    the image is blurred, which made the cube look ~10 % small / far (PnP depth above the f*L/side bound, HRL80 s_pnp
    1.13 x s_imu)."""
    tIdx0 = [int(np.argmin(np.sum((contour - tV) ** 2, 1))) for tV in verts]
    tOrder = np.argsort(tIdx0)                           # walk the vertices in CONTOUR order (the hull may run the other way)
    if len(set(tIdx0)) < 6:
        return verts
    tSharp = sharpenOrdered(contour, verts[tOrder], [tIdx0[tJ] for tJ in tOrder], keep)
    tOut = np.empty_like(verts)
    tOut[tOrder] = tSharp
    return tOut


def sharpenOrdered(contour, verts, idx, keep):
    """sharpen with the 6 vertices already in contour order (idx = their contour indices)"""
    tLines = []
    for tJ in range(6):
        tA, tB = idx[tJ], idx[(tJ + 1) % 6]
        tSeg = contour[tA:tB + 1] if tA <= tB else np.vstack([contour[tA:], contour[:tB + 1]])
        tM = len(tSeg)
        tSeg = tSeg[int(tM * keep[0]):max(int(tM * keep[1]), int(tM * keep[0]) + 2)]
        if len(tSeg) < 2:
            return verts
        tD = tSeg - tSeg.mean(0)
        _, _, tVt = np.linalg.svd(tD, full_matrices=False)
        tLines.append((tSeg.mean(0), tVt[0]))
    tOut = []
    for tJ in range(6):                                  # vertex j joins side j-1 and side j
        (tP1, tD1), (tP2, tD2) = tLines[tJ - 1], tLines[tJ]
        tA = np.array([tD1, -tD2]).T
        if abs(np.linalg.det(tA)) < PARALLEL_DET:
            tOut.append(verts[tJ])
            continue
        tT = np.linalg.solve(tA, tP2 - tP1)
        tQ = tP1 + tT[0] * tD1
        tMaxShift = SHARPEN_MAX_SHIFT_FRAC * np.linalg.norm(verts[(tJ + 1) % 6] - verts[tJ - 1])
        tOut.append(tQ if np.linalg.norm(tQ - verts[tJ]) < tMaxShift else verts[tJ])
    return np.array(tOut)


def occludedBelow(bgr, quad, band=OCCLUSION_BAND_PX, dark=OCCLUSION_DARK) -> bool:
    """True if the image band just below the quad's lowest edge is dark: the black gripper jaw covers the face there, so that
    edge is the jaw's outline, not the cube's (measured: these near frames moved the cube-centre residual 0.5 -> 4 cm)."""
    import cv2
    tLow = quad[np.argsort(quad[:, 1])[-2:]]                 # the two lowest vertices
    tGrey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tH, tW = tGrey.shape
    tXs = np.linspace(tLow[0, 0], tLow[1, 0], OCCLUSION_SAMPLES)
    tYs = np.linspace(tLow[0, 1], tLow[1, 1], OCCLUSION_SAMPLES) + band / 2 + OCCLUSION_OFFSET_PX
    tValues = [tGrey[int(min(tH - 1, max(0, tY + tD))), int(min(tW - 1, max(0, tX)))]
               for tX, tY in zip(tXs, tYs) for tD in range(-band // 2, band // 2)]
    return float(np.median(tValues)) < dark


def pnpFace(quad, p, edge=CUBE_EDGE_M):
    """one visible square face: solvePnP on the face (z = 0 plane, 4 cyclic x 2 directions), cube centre = face centre
    pushed half an edge along the face normal AWAY from the camera.  -> (t_centre [m], rvec, rms px, "1face") or None"""
    import cv2
    tSquare = np.array([(-1, 1, 0), (1, 1, 0), (1, -1, 0), (-1, -1, 0)], float) * edge / 2
    tBest = None
    for tDirection in (1, -1):
        for tShift in range(4):
            tObj = np.roll(tSquare[::tDirection], tShift, axis=0)
            tOk, tRvec, tTvec = cv2.solvePnP(tObj, quad, p, None, flags=cv2.SOLVEPNP_IPPE)
            if not tOk or tTvec.ravel()[2] <= 0:
                continue
            tRvec, tTvec = cv2.solvePnPRefineLM(tObj, quad, p, None, tRvec, tTvec)
            tProj, _ = cv2.projectPoints(tObj, tRvec, tTvec, p, None)
            tErr = float(np.sqrt(np.mean(np.sum((tProj.reshape(-1, 2) - quad) ** 2, 1))))
            tR, _ = cv2.Rodrigues(tRvec)
            tF = tTvec.ravel()
            tN = tR[:, 2]
            if float(tN @ (-tF)) < 0:
                tN = -tN                                     # outward normal points toward the camera
            # ONE face only when it is seen nearly head-on: an oblique view shows 2-3 faces whose silhouette can still reduce to 4
            # vertices; treating that as a single square biased s by -22 % on HRA ep 1 / 2 (multi-face 0.55 vs "1face" 0.43)
            if float(tN @ (-tF)) / np.linalg.norm(tF) < np.cos(np.radians(FACE_MAX_VIEW_DEG)):
                continue
            tCentre = tF - tN * edge / 2
            if tBest is None or tErr < tBest[2]:
                tBest = (tCentre, tRvec.ravel().copy(), tErr, "1face")
    return tBest


def pnpCube(hexPts, p, edge=CUBE_EDGE_M):
    """best of {3-face, 2-face} x 6 cyclic shifts x 2 directions; a candidate is admissible only if the faces that face
    the camera are exactly the model's visible faces (cheirality / visibility).  -> (t [m], rvec, rms px, model) or None.
    A 4-corner silhouette goes to pnpFace."""
    if len(hexPts) == 4:
        return pnpFace(hexPts, p, edge)
    import cv2
    tBest = None
    for tName, (tSil, tVisible) in MODELS.items():
        for tDirection in (1, -1):
            tSeq = tSil[::tDirection]
            for tShift in range(6):
                tObj = np.roll(tSeq, tShift, axis=0) * edge / 2
                tOk, tRvec, tTvec = cv2.solvePnP(tObj, hexPts, p, None, flags=cv2.SOLVEPNP_SQPNP)
                if not tOk:
                    continue
                tRvec, tTvec = cv2.solvePnPRefineLM(tObj, hexPts, p, None, tRvec, tTvec)
                if tTvec.ravel()[2] <= 0:
                    continue
                tR, _ = cv2.Rodrigues(tRvec)
                tCam = -tR.T @ tTvec.ravel()                     # camera centre in the cube frame
                tFacing = {tF for tF in _FACES if float(np.dot(tF, tCam)) > edge / 2 * (1 + VISIBLE_FACE_MARGIN)}
                if tFacing != tVisible:
                    continue
                tProj, _ = cv2.projectPoints(tObj, tRvec, tTvec, p, None)
                tErr = float(np.sqrt(np.mean(np.sum((tProj.reshape(-1, 2) - hexPts) ** 2, 1))))
                if tBest is None or tErr < tBest[2]:
                    tBest = (tTvec.ravel().copy(), tRvec.ravel().copy(), tErr, tName)
    return tBest


def quatXyzwToR(q) -> np.ndarray:
    """xyzw quaternion (ss1.csv column order) -> rotation matrix"""
    tX, tY, tZ, tW = q
    return np.array([[1 - 2 * (tY * tY + tZ * tZ), 2 * (tX * tY - tZ * tW), 2 * (tX * tZ + tY * tW)],
                     [2 * (tX * tY + tZ * tW), 1 - 2 * (tX * tX + tZ * tZ), 2 * (tY * tZ - tX * tW)],
                     [2 * (tX * tZ - tY * tW), 2 * (tY * tZ + tX * tW), 1 - 2 * (tX * tX + tY * tY)]])


def solveK(p, u) -> float:
    """k minimising sum |p_i + k u_i - c|^2 over k, c"""
    tDp, tDu = p - p.mean(0), u - u.mean(0)
    tDen = float(np.sum(tDu * tDu))
    return -float(np.sum(tDp * tDu)) / tDen if tDen > 0 else np.nan


def fitScale(p, u, *, trimIter=TRIM_ITERS, rng=None) -> dict:
    """k with MAD trimming of the cube-centre residuals, s = 1 / k, bootstrap p16 / p84 of s, residual RMS in metres"""
    tKeep = np.ones(len(p), bool)
    tK = np.nan
    for _ in range(trimIter):
        tK = solveK(p[tKeep], u[tKeep])
        tC = (p + tK * u)[tKeep].mean(0)
        tResid = np.linalg.norm(p + tK * u - tC, axis=1) / max(abs(tK), K_EPS)          # metres
        tMed = np.median(tResid[tKeep])
        tMad = np.median(np.abs(tResid[tKeep] - tMed)) + MAD_EPS
        tNew = tResid <= tMed + TRIM_MAD_K * MAD_TO_SIGMA * tMad
        if tNew.sum() < MIN_FRAMES or np.array_equal(tNew, tKeep):
            break
        tKeep = tNew
    tK = solveK(p[tKeep], u[tKeep])
    tC = (p + tK * u)[tKeep].mean(0)
    tResid = np.linalg.norm(p + tK * u - tC, axis=1) / max(abs(tK), K_EPS)
    tRng = rng or np.random.default_rng(0)
    tIdx = np.flatnonzero(tKeep)
    tBoots = []
    for _ in range(BOOTSTRAP_N):
        tB = tRng.choice(tIdx, len(tIdx))
        tKb = solveK(p[tB], u[tB])
        if np.isfinite(tKb) and tKb > 0:
            tBoots.append(1 / tKb)
    return dict(k=tK, s=1 / tK if tK > 0 else np.nan, keep=tKeep, centre_resid_m=float(np.sqrt(np.mean(tResid[tKeep] ** 2))),
                s_p16=float(np.percentile(tBoots, BOOTSTRAP_PCT[0])) if tBoots else np.nan,
                s_p84=float(np.percentile(tBoots, BOOTSTRAP_PCT[1])) if tBoots else np.nan)


def episodePnpScale(video, setting, ss1Csv, *, edge=CUBE_EDGE_M, stride=1, maxFrac: float = 1.0) -> dict:
    """one episode -> QC dict with s_pnp.  maxFrac < 1: only the first fraction of the video (validation on stacking takes,
    where the cube is static only until grasped)"""
    import cv2
    import pandas as pd
    tK, tD, tSize = readSetting(setting)
    tP, tUndistort = undistorter(tK, tD, tSize)
    tSs = pd.read_csv(ss1Csv).set_index("frame_idx")
    tCap = cv2.VideoCapture(str(video))
    tI = -1
    tFrames = []
    tTotal = int(tCap.get(cv2.CAP_PROP_FRAME_COUNT))
    tIMax = int(tTotal * maxFrac)
    while True:
        tOk, tImg = tCap.read()
        tI += 1
        if not tOk:
            break
        if tI > tIMax:
            break
        if tI % stride or tI not in tSs.index or str(tSs.loc[tI, "is_lost"]).lower() == "true":
            continue
        tUnd = tUndistort(tImg)
        tMask = cubeBlob(tUnd)
        if tMask is None:
            continue
        tHex, tSolidity = hexagon(tMask)
        if tHex is None or tSolidity < MIN_SOLIDITY:
            tFrames.append(dict(i=tI, ok=False, why="no_hexagon" if tHex is None else "solidity", area=int(tMask.sum())))
            continue
        if len(tHex) == 4 and occludedBelow(tUnd, tHex):
            tFrames.append(dict(i=tI, ok=False, why="occluded", area=int(tMask.sum())))
            continue
        tPnp = pnpCube(tHex, tP, edge)
        if tPnp is None:
            tFrames.append(dict(i=tI, ok=False, why="pnp", area=int(tMask.sum())))
            continue
        tT, _, tErr, tModel = tPnp
        tFrames.append(dict(i=tI, ok=tErr <= MAX_REPROJ_PX, why="" if tErr <= MAX_REPROJ_PX else "reproj", area=int(tMask.sum()),
                            reproj_px=tErr, model=tModel, dist_m=float(np.linalg.norm(tT)), t=tT.tolist()))
    tCap.release()
    tGood = [tF for tF in tFrames if tF["ok"]]
    tOut = dict(n_candidate_frames=len(tFrames), n_valid_pnp=len(tGood), cube_edge_m=edge, frames=tFrames,
                reproj_px_p50=float(np.median([tF["reproj_px"] for tF in tGood])) if tGood else None,
                reject_counts={tW: sum(1 for tF in tFrames if not tF["ok"] and tF["why"] == tW)
                               for tW in sorted(set(tF["why"] for tF in tFrames if not tF["ok"]))})
    if len(tGood) < MIN_FRAMES:
        return dict(tOut, valid=False, reason=f"only {len(tGood)} valid PnP frames (< {MIN_FRAMES})")
    tPw = np.array([[tSs.loc[tF["i"], tC] for tC in ("x", "y", "z")] for tF in tGood], float)
    tR = np.array([quatXyzwToR([tSs.loc[tF["i"], tC] for tC in ("q_x", "q_y", "q_z", "q_w")]) for tF in tGood])
    tU = np.einsum("nij,nj->ni", tR, np.array([tF["t"] for tF in tGood]))
    tFit = fitScale(tPw, tU)
    tSpread = (tFit["s_p84"] - tFit["s_p16"]) / 2 / tFit["s"] if np.isfinite(tFit["s"]) else np.inf
    tOut.update(s_pnp=tFit["s"], s_p16=tFit["s_p16"], s_p84=tFit["s_p84"], rel_spread=float(tSpread),
                centre_resid_m=tFit["centre_resid_m"], n_used=int(tFit["keep"].sum()),
                dist_m_range=[float(min(tF["dist_m"] for tF in tGood)), float(max(tF["dist_m"] for tF in tGood))])
    tWhy = []
    if not np.isfinite(tFit["s"]) or tFit["s"] <= 0:
        tWhy.append("non-positive scale")
    if tSpread > MAX_REL_SPREAD:
        tWhy.append(f"bootstrap spread {tSpread:.2f} > {MAX_REL_SPREAD}")
    if tFit["centre_resid_m"] > MAX_CENTRE_RESID_M:
        tWhy.append(f"cube-centre residual {tFit['centre_resid_m'] * CM_PER_M:.1f} cm > {MAX_CENTRE_RESID_M * CM_PER_M:.0f} cm")
    return dict(tOut, valid=not tWhy, reason="; ".join(tWhy))
