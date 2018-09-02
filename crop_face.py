import os
import numpy as np
import cv2

#ROOT = "C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\"
#os.chdir(ROOT)

# 2.1 Face detection and cropping
# This function is put in separate script for ease of development


# Load cv2 face detection module
xml_PATH="C:\\Users\\peanut\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(xml_PATH)

# An example for debugging
#img_path = ROOT + train_crop\\Anger\\S010_004_00000019.jpg'

def crop_face (img_path,dst='Same',ret=False):
    # img_path: Path to jpg image.
    # dst: Destination to save cropped image. 
    #   If 'same', will perform in-place operation. Or else, saved in relative location to img_path 
    # ret: If True, return np array instead of jpg image. Np aray is required for use in substract_neutral.py
    
    folder_path, filename = os.path.split(img_path)        
    if dst == 'Same':
        dst_path = img_path
    else:
        dst_path = os.path.join(folder_path,dst)        
    
    img = cv2.imread(img_path,0)    
    faces = face_cascade.detectMultiScale(img, 1.05, 5)
    x,y,w,h = faces[0]
    img_crop = img[y:y+h,x:x+w]
    img_crop = cv2.resize(img_crop, (224,224))
           
    if ret: 
        return img_crop
    else:
        cv2.imwrite(dst_path,img_crop)

def run_crop_face (ROOT):
    # Make folders
    try:
        for i in range(1,8):
            os.makedirs('train_crop\\' + emo_dict[i])
        os.makedirs('test_crop')
    except:
        print ('Folders already exist')
    
    # cropping test data
    TEST_PATH = ROOT + 'test\\'
    TEST_CROP_PATH = ROOT + 'test_crop\\'
    
    for file in os.listdir(TEST_PATH):
        crop_face (TEST_PATH + file,TEST_CROP_PATH + file)
    
    
    # cropping training data
    TRAIN_PATH = ROOT + 'train\\'
    TRAIN_CROP_PATH = ROOT + 'train_crop\\'
    
    for folder in os.listdir(TRAIN_PATH):
        
        for file in os.listdir(TRAIN_PATH + folder):
            crop_face(TRAIN_PATH + folder + '\\' + file, TRAIN_CROP_PATH + folder + '\\' + file,)
        

emo_dict = {0:'Neutral',
            1:'Anger',
            2:'Contempt',
            3:'Disgust',
            4:'Fear',
            5:'Happy',
            6:'Sadness',
            7:'Surprise'
            }        

#for debugging
#new_img = cv2.imread(r'C:\Users\peanut\Dropbox\KE5108 DEVELOPING INTELLIGENT SYSTEMS\CA3\train\Surprise\S076_001_00000017.jpg',0)
#new_img = cv2.imread(r'C:\Users\peanut\Dropbox\KE5108 DEVELOPING INTELLIGENT SYSTEMS\CA3\train\Anger\S010_004_00000019.jpg',0)
#faces = face_cascade.detectMultiScale(new_img, 1.3, 1)
#x,y,w,h = faces[0]
#cv2.rectangle(new_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#cv2.imshow('img',img_crop)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

