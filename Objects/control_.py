import os
import sys
import cv2
import numpy as np
import os.path as osp


class Control:
    def __init__(self):
        # basic infos
        self._time_process = None
        self._time_process_emotionDetection = None
        self._time_process_eyegazeEstimation = None
        self._frame_idx = 0
        self._count_face = 0
        # running_control


    def process(self, config):
        self._time_process = config.time_process
        self._time_process_emotionDetection = config.time_process
        self._time_process_eyegazeEstimation = config.time_process

