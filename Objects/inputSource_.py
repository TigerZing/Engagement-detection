import os
import sys
import cv2
import numpy as np
import os.path as osp
import pafy

class InputSource:
    def __init__(self):
        # basic infos
        self.path = None
        self.type = None # return "video|streaming|online"
        self.width = None
        self.height = None
        self.fps = None
        self.length = None

    def process(self, config):
        # Check input_video is recored video or streaming video
        if type(config.source_video) is int:
            self.type = "streaming"
        elif type(config.source_video) is str:
            if config.source_video.find("http")!=-1:
                video = pafy.new(config.source_video)
                best  = video.getbest(preftype="mp4")
                self.type = "online"
                self.path = best.url
            else:
                self.type = "video"
        else:
            print("[Error] The type of input video is unknown!")
        # process
        video = cv2.VideoCapture(config.source_video)
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        self.height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height
        if self.path is None:
            self.path = config.source_video 
        if self.type == "video":
            self.length = video.get(cv2.CAP_PROP_FRAME_COUNT)
        elif self.type == "online":
            self.fps = 30
            video = pafy.new(config.source_video)
            best  = video.getbest(preftype="mp4")
            self.width  = best.dimensions[0]   # float `width`
            self.height = best.dimensions[1]   # float `height



