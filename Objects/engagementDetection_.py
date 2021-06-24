import os
import sys
import cv2
import numpy as np
import os.path as osp
import tensorflow as tf


def detect_engagement(emotion, attention):
    #print("emotion {} attention {}".format(emotion, attention))
    engagement_level = None
    if attention is None:
        engagement_level = "medium engagement"
    elif attention == "focus":
        if emotion == "Disgusted" or emotion == "Surprised":
            engagement_level = "strong engagement"
        elif emotion == "Angry" or emotion == "Fearful":
            engagement_level = "high engagement"
        elif emotion == "Sad" or emotion == "Happy":
            engagement_level = "medium engagement"
        elif emotion == "Neutral":
            engagement_level = "low engagement"
    elif attention == "distracted":
        engagement_level = "disengagement"
    else : 
        print("can not detect engagement!")
        sys.exit()
    if engagement_level == None:
        pass
    return engagement_level