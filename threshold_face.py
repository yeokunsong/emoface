import os
import numpy as np
import cv2

#ROOT = "C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\"
#os.chdir(ROOT)

# 2.2 Thresholding face for noise reduction
# This function is put in separate script for ease of development

def threshold_face (img_path,dst='Same',flag='thre'):
    # img_path: Path to jpg image.
    # dst: Destination to save cropped image. 
    #   If 'same', will perform in-place operation. Or else, saved in relative location to img_path 
    # flag: Determine type of thresholding.
    #   If 'thre', return binary threshold of jpg image. 
    #   If 'ath', return adaptive threshold of jpg image. 
 
    folder_path, filename = os.path.split(img_path)        
    if dst == 'Same':
        dst_path = img_path
    else:
        dst_path = os.path.join(folder_path,dst)
    
    img = cv2.imread(img_path,0)
    retval, threshold = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    ath = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,5)
        
    if flag == 'thre':
        return cv2.imwrite(dst_path,threshold)
    elif flag == 'ath':
        return cv2.imwrite(dst_path,ath)
    
def run_threshold_face (ROOT):

    # Make folders
    try:
        os.makedirs('test_threshold')
        for i in range(1,8):
            os.makedirs('train_threshold\\' + emo_dict[i])
    except:
        print ('Folders already exist')        

    # Thresholding testing data
    TEST_CROP_PATH = ROOT + 'test_crop\\'
    TEST_THRESHOLD_PATH = ROOT + 'test_threshold\\'
    
    for file in os.listdir(TEST_CROP_PATH):
        threshold_face (TEST_CROP_PATH + file,TEST_THRESHOLD_PATH + file,flag='ath')
    
    
    # Thresholding training data
    TRAIN_CROP_PATH = ROOT + 'train_crop\\'
    TRAIN_THRESHOLD_PATH = ROOT + 'train_threshold\\'
    
    for folder in os.listdir(TRAIN_CROP_PATH):
        
        for file in os.listdir(TRAIN_CROP_PATH + folder):
            threshold_face(TRAIN_CROP_PATH + folder + '\\' + file, TRAIN_THRESHOLD_PATH + folder + '\\' + file,flag='ath')
            
emo_dict = {0:'Neutral',
            1:'Anger',
            2:'Contempt',
            3:'Disgust',
            4:'Fear',
            5:'Happy',
            6:'Sadness',
            7:'Surprise'
                }