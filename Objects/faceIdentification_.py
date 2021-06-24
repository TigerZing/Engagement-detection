import os
import sys
import cv2
import numpy as np
import os.path as osp
import tensorflow as tf
from Objects import student_

def caculate_similarity(face, student_list):
    sift = cv2.xfeatures2d.SIFT_create()
    similarity_all = []
    for idx, student in enumerate(student_list):
        #face_person = cv2.imread(osp.join(config.face_data_dirPath ,person.id_student+"/"+person.face_id+".png"))[:,:,0]
        face_student = student._face_region
        #flag = 0 if face.shape[0]+face.shape[1]<face_student.shape[0]+face_student.shape[1] else 1 # flag =0 => face
        # -----
        keypoints_face, descriptors_face = sift.detectAndCompute(face, None) 
        keypoints_face_student, descriptors_face_student = sift.detectAndCompute(face_student, None)
        #cv2.imwrite("keypoints_face.png",cv2.drawKeypoints(face, keypoints_face, None, (255, 0, 255)))
        #cv2.imwrite("keypoints_face_person.png",cv2.drawKeypoints(face_student, keypoints_face_person, None, (255, 0, 255))) 
        if descriptors_face is None or descriptors_face_student is None:
            return False, None
        # feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_face, descriptors_face_student)
        if matches ==[]:
            return False, None
        else:
            matches = sorted(matches, key = lambda x:x.distance)
            total_matches = 5 if len(matches) >=5 else len(matches)
            sum_simalarity = 0
            for matche in matches[:int(total_matches)-1]:
                sum_simalarity+=matche.distance
            similarity_all.append(int(sum_simalarity/total_matches))
        #img3 = cv2.drawMatches(face, keypoints_face, face_person, keypoints_face_person, matches[:min(len(keypoints_face),len(keypoints_face_person))], None, flags=2)
        #cv2.imwrite("final.png",img3)
        isSimilar = False
        if min(similarity_all) >int(1400):
            return isSimilar, None
        return True, student_list[similarity_all.index(min(similarity_all))]

def identify_face(control, face, face_coord, student_list):

    threshold=  0.8*(max(face_coord[2],face_coord[3])) # set distance
    face_point = ((face_coord[0]+face_coord[2])//2,(face_coord[1]+face_coord[3])//2)
    min_distance = None
    findedstudent = None
    #isSimilar, person = caculate_similarity(config, face[:,:,0], control.list_student)
    for idx, student in enumerate(student_list):
        distance = abs(face_point[0]-student._face_point[0])+abs(face_point[1]-student._face_point[1])
        if idx == 0:
            min_distance = distance
            findedstudent = student
        elif distance< min_distance:
            min_distance = distance
            findedstudent = student

    if min_distance>int(threshold):
        isSimilar, student = caculate_similarity(face, student_list)
        if isSimilar:
            student._face_point = face_point
            student._face_region = face
            return student
        else:
            new_student = student_.Student(str(control._count_face), str(control._count_face))
            #id_ = osp.splitext(osp.basename(config.video_path))[0]+"_"+"%06d"%control.index_frame+"_"+"%06d"%control.index_face # ID of face
            new_student._face_point = face_point
            new_student._face_region = face
            student_list.append(new_student)
            return new_student
    else:
        findedstudent._face_point = face_point
        findedstudent._face_region = face
    return findedstudent
