import os
import sys
import cv2
import dlib
import numpy as np
import os.path as osp
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import operator

def analyseface( img, landMarksDetector, dets, quality=1, offset=(0,0)):
    result=[]
    for k, face_ord in enumerate(dets):
        #d = [(face_ord[0],face_ord[1]),(face_ord[0]+face_ord[2],face_ord[1]+face_ord[3])]
        d = dlib.rectangle(face_ord[0],face_ord[1],face_ord[0]+face_ord[2],face_ord[1]+face_ord[3])
        instantFacePOI = np.zeros((7,2),dtype=np.float32)
        eyeCorners=np.zeros((2,2,2),dtype=np.float32)
        # Get the landmarks/parts for the face in box d.
        shape = landMarksDetector(np.array(img), d)
        #oreille droite
        instantFacePOI[0][0]=shape.part(0).x+offset[0];
        instantFacePOI[0][1]=shape.part(0).y+offset[1];
        #oreille gauche
        instantFacePOI[1][0]=shape.part(16).x+offset[0];
        instantFacePOI[1][1]=shape.part(16).y+offset[1];
        #nez
        instantFacePOI[2][0]=shape.part(30).x+offset[0];
        instantFacePOI[2][1]=shape.part(30).y+offset[1];
        #bouche gauche
        instantFacePOI[3][0]=shape.part(48).x+offset[0];
        instantFacePOI[3][1]=shape.part(48).y+offset[1];
        #bouche droite
        instantFacePOI[4][0]=shape.part(54).x+offset[0];
        instantFacePOI[4][1]=shape.part(54).y+offset[1];

        leftEyeX=0
        leftEyeY=0
        for i in range(36, 42):
            leftEyeX+=shape.part(i).x
            leftEyeY+=shape.part(i).y
        leftEyeX=int(leftEyeX/6.0)
        leftEyeY=int(leftEyeY/6.0)
        eyeCorners[0][0]=[shape.part(36).x+offset[0],shape.part(36).y+offset[1]]
        eyeCorners[0][1]=[shape.part(39).x+offset[0],shape.part(39).y+offset[1]]

        instantFacePOI[5][0]=leftEyeX+offset[0];
        instantFacePOI[5][1]=leftEyeY+offset[1];

        rightEyeX=0
        rightEyeY=0
        for i in range(42, 48):
            rightEyeX+=shape.part(i).x
            rightEyeY+=shape.part(i).y
        rightEyeX=int(rightEyeX/6.0)
        rightEyeY=int(rightEyeY/6.0)
        eyeCorners[1][0]=[shape.part(42).x+offset[0],shape.part(42).y+offset[1]]
        eyeCorners[1][1]=[shape.part(45).x+offset[0],shape.part(45).y+offset[1]]
        instantFacePOI[6][0]=rightEyeX+offset[0];
        instantFacePOI[6][1]=rightEyeY+offset[1];
        data=[instantFacePOI, (int(d.left()+offset[0]),int(d.top()+offset[1]),int(d.right()+offset[0]),int(d.bottom()+offset[1])),eyeCorners]
        result.append(data)
    return result

def analyse_face(img, landMarksDetector, face_ord, quality=1, offset=(0,0)):
    d = dlib.rectangle(face_ord[0],face_ord[1],face_ord[0]+face_ord[2],face_ord[1]+face_ord[3])
    instantFacePOI = np.zeros((7,2),dtype=np.float32)
    eyeCorners=np.zeros((2,2,2),dtype=np.float32)
    # Get the landmarks/parts for the face in box d.
    shape = landMarksDetector(np.array(img), d)
    #oreille droite
    instantFacePOI[0][0]=shape.part(0).x+offset[0];
    instantFacePOI[0][1]=shape.part(0).y+offset[1];
    #oreille gauche
    instantFacePOI[1][0]=shape.part(16).x+offset[0];
    instantFacePOI[1][1]=shape.part(16).y+offset[1];
    #nez
    instantFacePOI[2][0]=shape.part(30).x+offset[0];
    instantFacePOI[2][1]=shape.part(30).y+offset[1];
    #bouche gauche
    instantFacePOI[3][0]=shape.part(48).x+offset[0];
    instantFacePOI[3][1]=shape.part(48).y+offset[1];
    #bouche droite
    instantFacePOI[4][0]=shape.part(54).x+offset[0];
    instantFacePOI[4][1]=shape.part(54).y+offset[1];
    leftEyeX=0
    leftEyeY=0
    for i in range(36, 42):
        leftEyeX+=shape.part(i).x
        leftEyeY+=shape.part(i).y
    leftEyeX=int(leftEyeX/6.0)
    leftEyeY=int(leftEyeY/6.0)
    eyeCorners[0][0]=[shape.part(36).x+offset[0],shape.part(36).y+offset[1]]
    eyeCorners[0][1]=[shape.part(39).x+offset[0],shape.part(39).y+offset[1]]
    instantFacePOI[5][0]=leftEyeX+offset[0];
    instantFacePOI[5][1]=leftEyeY+offset[1];
    rightEyeX=0
    rightEyeY=0
    for i in range(42, 48):
        rightEyeX+=shape.part(i).x
        rightEyeY+=shape.part(i).y
    rightEyeX=int(rightEyeX/6.0)
    rightEyeY=int(rightEyeY/6.0)
    eyeCorners[1][0]=[shape.part(42).x+offset[0],shape.part(42).y+offset[1]]
    eyeCorners[1][1]=[shape.part(45).x+offset[0],shape.part(45).y+offset[1]]
    instantFacePOI[6][0]=rightEyeX+offset[0];
    instantFacePOI[6][1]=rightEyeY+offset[1];
    data=[instantFacePOI, (int(d.left()+offset[0]),int(d.top()+offset[1]),int(d.right()+offset[0]),int(d.bottom()+offset[1])),eyeCorners]
    return data


class Face_LandMarkDetection:
    def __init__(self):
        # basic infos
        self.type = None # haarcascade|mtcnn
        self.weight_path = None
        self.path = None

    def process(self, config):
        # Check input_video is recored video or streaming video
        if config.type_landMaskDetection == "68_landmarks":
            self.type = "68_landmarks"
            self.path = osp.join(config.algorithm_path,self.type)
            self.weight_path = osp.join(self.path , 'shape_predictor_68_face_landmarks.dat')
        else:
            print("[Error] The type of Face_LandMarkDetection is unknown!")
        # process
    
    def intialize(self):
        sys.path.append(self.path)
        if self.type == "68_landmarks":
            return dlib.shape_predictor(self.weight_path)
    
    def detect_landMarks(self, landMarksDetector, face, gray):
        face_data = None
        if self.type == "68_landmarks":
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_data = analyse_face(gray, landMarksDetector, face, 0)
        # convert to standard
        return face_data



