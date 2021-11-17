import os
import sys
import cv2
import numpy as np
import os.path as osp
import tensorflow as tf

class Frame:
    def __init__(self,_frame_id):
        #basic infos
        self._frame_id = _frame_id
        # analytic results
        self._emotion = {"Angry": 0, "Disgusted": 0, "Fearful" :0, "Happy":0, "Neutral":0, "Sad":0, "Surprised":0}
        self._attention = {"focused":0, "distracted":0}
        self._engagement_level_mapping_method = {"strong engagement":0, "medium engagement":0, "high engagement":0, "low engagement": 0, "disengagement":0}
        self._engagement_level_KES_method = {"disengagement":0,"engagement":0,"high engagement":0}
    
    def _intialize(self, frame_id):
        self._frame_id = frame_id