import os
import sys
import cv2
import numpy as np
import os.path as osp
import tensorflow as tf


class Student:
    def __init__(self, _id, _name):
        # basic infos
        self._id = _id
        self._name = _name
        # status
        self._face_region = None
        self._face_point = None
        self._status_eyeGaze_count = 0
        self._max_distance = 0 #eyeGaze line
        self._total_distance = 0 #eyeGaze line

        # analytic results
        self._emotion = {"Angry": 0, "Disgusted": 0, "Fearful" :0, "Happy":0, "Neutral":0, "Sad":0, "Surprised":0}
        self._attention = {"focused":0, "distracted":0}
        self._engagement_level = {"strong engagement":0, "medium engagement":0, "high engagement":0, "low engagement": 0, "disengagement":0}
        # save analytic results
        self._analytic_results = []
    
    def _reset_analytic_results(self):
        self._emotion = {"Angry": 0, "Disgusted": 0, "Fearful" :0, "Happy":0, "Neutral":0, "Sad":0, "Surprised":0}
        self._attention = {"focused":0, "distracted":0}
        self._engagement_level = {"strong engagement":0, "medium engagement":0, "high engagement":0, "low engagement": 0, "disengagement":0}

    def _intialize(self, id, name):
        self.id = id
        self.name = name
        # analytic results
        
    
    def _add_dict(self,engagement_level, emotion, attention, faceName_file, face_coord, frame_idx):
        dict_ = {'emotion': emotion, 
                'engagement_level': engagement_level, 
                'attention': attention,
                'faceName_file':faceName_file,
                'face_coord':face_coord,
                'frame_idx':frame_idx}
        self._analytic_results.append(dict_)


