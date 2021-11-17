import cv2 
import numpy as np 
import face_recognition
import os
from datetime import datetime
import csv
import glob
import os.path as osp
#LOAD IMAGE AND CONVERT TO RGB IMAGE

def check_margin_boudingbox(img, bouding_box, size):
    height, width, _ = img.shape
    min_ = size  
    margin_top = bouding_box[1]
    margin_left = bouding_box[0]
    margin_bottom = height - (bouding_box[1] + bouding_box[3])
    margin_right = width - (bouding_box[0] + bouding_box[2])
    min_ = min(margin_top, margin_left, margin_bottom, margin_right)
    if size < min_:
        min_ = size
    return min_




def findEncoding(images, faceDetector_Config, faceDetector):
    encodeList = []
    faces = []
    for img in images:
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #img= cv2.resize(img, (400,200), interpolation = cv2.INTER_AREA)
        #print(img.shape)
        #face_loc = face_recognition.face_locations(img)
        detected_faces = faceDetector_Config.detect_faces(faceDetector, img)
        if detected_faces !=[]:
            detected_faces = detected_faces[0]['box']
        else:
            continue
        [x,y,w,h] = detected_faces
        min_ = check_margin_boudingbox(img, detected_faces, 10)
        face_ROI = cv2.cvtColor(img[(y-min_):(y+h+min_), (x-min_):(x+w+min_)],cv2.COLOR_BGR2RGB)
        faces.append(face_ROI)
        face_loc = [tuple([detected_faces[1],detected_faces[0]+detected_faces[2],detected_faces[1]+detected_faces[3],detected_faces[0]])] 
        encode = face_recognition.face_encodings(img, face_loc)[0]
        #print(encode)
        encodeList.append(encode)
    return encodeList, faces


def encode_face(img_paths, faceDetector_Config, faceDetector):
    images = []
    classNames = []
    ext = ['png', 'jpg', 'gif']    # Add image formats here
    file_paths = []
    [file_paths.extend(glob.glob(osp.join(img_paths, '*.' + e))) for e in ext]
    for path in file_paths:
        curImg = cv2.imread(path)
        images.append(curImg)
        classNames.append(os.path.splitext(osp.basename(path))[0])
    encodeFaceList, faces  = findEncoding(images, faceDetector_Config, faceDetector)
    print('Encoding Completed, the total images is:',len(encodeFaceList))
    return encodeFaceList, classNames, faces
   

def recognize_face(encodeFaceList, classNames, frame, face_coord):
    name = None
    #imgS = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #facesCurFrame = face_recognition.face_locations(imgS)
    #encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # turn on face_recognition

    face_CurrentFrame = [tuple([face_coord[1],face_coord[0]+face_coord[2],face_coord[1]+face_coord[3],face_coord[0]])]
    encodes_CurrentFrame = face_recognition.face_encodings(frame, face_CurrentFrame)
    matches = face_recognition.compare_faces(encodeFaceList, encodes_CurrentFrame[0])#,tolerance=0.4
    faceDis = face_recognition.face_distance(encodeFaceList, encodes_CurrentFrame[0])
    matchIndex = np.argmin(faceDis)
    if matches[matchIndex]:
        name = classNames[matchIndex]
    return name
