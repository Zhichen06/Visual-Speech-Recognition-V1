#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import cv2
import hydra
import torchvision
from pipelines.detectors.mediapipe.detector import LandmarksDetector
from pipelines.data.data_module import AVSRDataLoader


def save2vid(filename, vid, frames_per_second):
    import cv2
    import numpy as np
    import torch
    if isinstance(vid, torch.Tensor): vid = vid.cpu().numpy()
    if vid.ndim == 4 and vid.shape[1] == 3:
        vid = vid.transpose(0, 2, 3, 1)
    t, h, w, c = vid.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, float(frames_per_second), (w, h))
    for i in range(t):
        frame = vid[i].astype(np.uint8)
        if c == 3: frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    if cfg.detector == "mediapipe":
        from pipelines.detectors.mediapipe.detector import LandmarksDetector
        landmarks_detector = LandmarksDetector()
    if cfg.detector == "retinaface":
        from pipelines.detectors.retinaface.detector import LandmarksDetector
        landmarks_detector = LandmarksDetector()
    dataloader = AVSRDataLoader(modality="video", speed_rate=1, transform=False, detector=cfg.detector, convert_gray=False)
    landmarks = landmarks_detector(cfg.data_filename)
    data = dataloader.load_data(cfg.data_filename, landmarks)
    fps = cv2.VideoCapture(cfg.data_filename).get(cv2.CAP_PROP_FPS)
    save2vid(cfg.dst_filename, data, fps)
    print(f"The mouth images have been cropped and saved to {cfg.dst_filename}")


if __name__ == "__main__":
    main()
