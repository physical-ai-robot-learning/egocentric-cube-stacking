"""Processed episode files (spec 23): writer, reader and the optional row-rate video (was io/processed_writer.py).

One directory per episode: one .npy per array in ARRAYS + metadata.json (with array_sha256_16, the first 16 hex digits
of each .npy's sha256); one folder per camera with frame_index.npy (row -> source frame, -1 = none) and source.json.

Pipeline stage: output of convert.py / input of the validators, the dataset reader and the LeRobot exporter.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

ARRAYS = ("timestamps", "row_valid", "train_rows", "state", "state_prevrel", "state_prev_valid", "cart20", "action",
          "left_pose", "right_pose", "left_gripper", "right_gripper", "stage_id", "camera_valid")
SHA_HEX_DIGITS = 16
ROW_VIDEO_CRF = "18"           # H.264 quality of the row-rate preview video


def writeEpisode(outDir, ep: dict, overwrite: bool = False) -> pathlib.Path:
    """convertEpisode() result -> episode directory; FileExistsError if it exists and overwrite is False"""
    tOut = pathlib.Path(outDir)
    if tOut.exists() and not overwrite:
        raise FileExistsError(tOut)
    tOut.mkdir(parents=True, exist_ok=True)
    tSha = {}
    for tKey in ARRAYS:
        np.save(tOut / f"{tKey}.npy", ep[tKey])
        tSha[f"{tKey}.npy"] = hashlib.sha256((tOut / f"{tKey}.npy").read_bytes()).hexdigest()[:SHA_HEX_DIGITS]
    for tCam, tInfo in ep["cameras"].items():
        tCamDir = tOut / tCam
        tCamDir.mkdir(exist_ok=True)
        np.save(tCamDir / "frame_index.npy", tInfo["frame_index"])
        json.dump({tK: tV for tK, tV in tInfo.items() if tK != "frame_index"}, open(tCamDir / "source.json", "w"), indent=1)
    tMeta = dict(ep["metadata"])
    tMeta["array_sha256_16"] = tSha
    json.dump(tMeta, open(tOut / "metadata.json", "w"), indent=1)
    return tOut


def loadEpisode(epDir, mmap: bool = True) -> dict:
    """episode directory -> dict of arrays (memory-mapped by default) + metadata + cameras"""
    tDir = pathlib.Path(epDir)
    tMode = "r" if mmap else None
    tEp = {tKey: np.load(tDir / f"{tKey}.npy", mmap_mode=tMode) for tKey in ARRAYS}
    tEp["metadata"] = json.load(open(tDir / "metadata.json"))
    tEp["cameras"] = {}
    for tCam in tEp["metadata"].get("cameras", []):
        tEp["cameras"][tCam] = dict(frame_index=np.load(tDir / tCam / "frame_index.npy"),
                                    **json.load(open(tDir / tCam / "source.json")))
    return tEp


def encodeRowVideo(srcMp4, frameIndex, dstMp4, size: int = 224, fps: int = 15) -> None:
    """decode the source mp4 sequentially and write the selected frames (row order) as a size x size H.264 mp4;
    rows whose frame_index is -1 get a black frame (marked invalid in camera_valid)"""
    import av
    tWant = {}
    for tRow, tFrame in enumerate(frameIndex):
        if tFrame >= 0:
            tWant.setdefault(int(tFrame), []).append(tRow)
    tRows = [None] * len(frameIndex)
    with av.open(str(srcMp4)) as tContainer:
        for tI, tDecoded in enumerate(tContainer.decode(video=0)):
            if tI in tWant:
                tImg = np.asarray(tDecoded.to_image().resize((size, size)))
                for tRow in tWant[tI]:
                    tRows[tRow] = tImg
            if tI > max(tWant, default=-1):
                break
    tBlank = np.zeros((size, size, 3), np.uint8)
    with av.open(str(dstMp4), "w") as tOutput:
        tStream = tOutput.add_stream("libx264", rate=int(fps))
        tStream.width = size
        tStream.height = size
        tStream.pix_fmt = "yuv420p"
        tStream.options = {"crf": ROW_VIDEO_CRF}
        for tImg in tRows:
            for tPacket in tStream.encode(av.VideoFrame.from_ndarray(tBlank if tImg is None else tImg, format="rgb24")):
                tOutput.mux(tPacket)
        for tPacket in tStream.encode():
            tOutput.mux(tPacket)
