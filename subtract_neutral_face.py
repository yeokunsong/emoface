# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 22:27:38 2018

@author: peanut
"""

import os
import numpy as np
import cv2

from crop_face import crop_face
from data_extraction_preprocessing import *

#ROOT = "C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\"
#os.chdir(ROOT)

# 2.3 Substract neutral faces from expression faces to extract 'motion' feature
# This function is put in separate script for ease of development

IMAGE_SOURCE_PATH = 'C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\extended-cohn-kanade-images\\cohn-kanade-images\\'

# An example for debugging
img_path = ROOT + 'train_crop\\Anger\\S010_004_00000019.jpg'

def get_neutral_face_path (img_path):
    # img_path: Path to jpg image, of expression face.
    
    filename = os.path.split(img_path)[-1]
    
    try: # if img is from \\test_crop
        label,sub,sub_trial,num = filename.split('_')
    except: # if img is from \\train_crop
        sub,sub_trial,num = filename.split('_')
        
    nf_filename = '_'.join((sub,sub_trial,'00000001.png'))
    nf_path = os.path.join(IMAGE_SOURCE_PATH,sub,sub_trial,nf_filename)
    
    return nf_path


def subtract_neutral_face (img_path,dst='Same',flag='diff'):
    # img_path: Path to jpg image, of expression face.
    # dst: Destination to save cropped image. 
    #   If 'same', will perform in-place operation. Or else, saved in relative location to img_path 
    # flag: Determine type of thresholding.
    #   If 'diff', return difference between neutral and expression face.
    #   If 'thre', return binary threshold of the difference. 
    #   If 'ath', return adaptive threshold of the difference. 
    
    folder_path, filename = os.path.split(img_path)
        
    if dst == 'Same':
        dst_path = img_path
    else:
        dst_path = os.path.join(folder_path,dst)
    
    # Extract and resize neutral face
    nf_path = get_neutral_face_path(img_path)
    nf_img = cv2.imread(nf_path,0)
    nf_img_crop = crop_face(nf_path,ret=True) #return 3D np array
    nf_img_crop_rs = cv2.resize(nf_img_crop, (224,224))
    
    # Read and resize expression face
    img = cv2.imread(img_path,0)
    img_rs = cv2.resize(img, (224,224))
    
    # Find absolute difference
    diff = cv2.subtract(img_rs, nf_img_crop_rs)
    retval, threshold = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    ath = cv2.adaptiveThreshold(diff,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3)
    
    # Find relative difference
    #diff2 = img_rs - nf_img_crop_rs
    #retval2, threshold2 = cv2.threshold(diff2, 50, 255, cv2.THRESH_BINARY)
    #ath2 = cv2.adaptiveThreshold(diff2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3)
    
    if flag == 'diff':
        return cv2.imwrite(dst_path,diff)
    elif flag == 'thre':
        return cv2.imwrite(dst_path,threshold)
    elif flag == 'ath':
        return cv2.imwrite(dst_path,ath)

def run_subtract_neutral_face (ROOT):
    # Make folders
    try:
        for i in range(1,8):
            os.makedirs('train_subtract\\' + emo_dict[i])
        os.makedirs('test_subtract')
    except:
        print ('Folders already exist')
    
    # cropping test data
    TEST_CROP_PATH = ROOT + 'test_crop\\'
    TEST_SUBTRACT_PATH = ROOT + 'test_subtract\\'
    
    for file in os.listdir(TEST_CROP_PATH):
        subtract_neutral_face (TEST_CROP_PATH + file,TEST_SUBTRACT_PATH + file,flag='diff')
    
    
    # cropping training data
    TRAIN_CROP_PATH = ROOT + 'train_crop\\'
    TRAIN_SUBTRACT_PATH = ROOT + 'train_subtract\\'
    
    for folder in os.listdir(TRAIN_CROP_PATH):
        
        for file in os.listdir(TRAIN_CROP_PATH + folder):
            subtract_neutral_face(TRAIN_CROP_PATH + folder + '\\' + file, TRAIN_SUBTRACT_PATH + folder + '\\' + file,flag='diff')


emo_dict = {0:'Neutral',
            1:'Anger',
            2:'Contempt',
            3:'Disgust',
            4:'Fear',
            5:'Happy',
            6:'Sadness',
            7:'Surprise'
            }

#cv2.imshow('ath2',ath2)
#cv2.imshow('threshold2',threshold2)
#cv2.imshow('diff2',diff2)
#
#cv2.imshow('ath',ath)
#cv2.imshow('threshold',threshold)
#cv2.imshow('diff',diff)
#
#cv2.imshow('nt_img_crop',nf_img_crop_rs)
#cv2.imshow('img',img_rs)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()
            

