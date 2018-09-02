import os
import pandas as pd
import random

from glob import glob
from shutil import copy2
import cv2


# Path to root
ROOT = "C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\23434\\"
ROOT = "C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\"

# Dataset has both labelled and unlabelled images
# http://www.consortium.ri.cmu.edu/data/ck/

# Emotions are labelled with integer in original dataset
# http://www.consortium.ri.cmu.edu/data/ck/CK+/ReadMeCohnKanadeDatabase_website.txt
emo_dict = {0:'Neutral',
            1:'Anger',
            2:'Contempt',
            3:'Disgust',
            4:'Fear',
            5:'Happy',
            6:'Sadness',
            7:'Surprise'
                }

# 1. Extract only images with labelled emotions 
# Path to folder of labels from downloaded dataset
LABEL_PATH = "C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\Emotion_labels\\Emotion\\"

# 1.1 Extract labels
# Directory structure is ./subject/subject_trial/[.txt]
# Unlabelled data will have no .txt file
def extract_label_from_CK_plus ():
    sub_list = os.listdir(LABEL_PATH)
    img_label = []
    
    for sub in sub_list:
        print (sub)
        
        for sub_trial in os.listdir(LABEL_PATH + sub):
            print (sub_trial)
            
            for emo in os.listdir(LABEL_PATH + sub + '\\' + sub_trial):
                print (emo)
                
                f = open (LABEL_PATH + sub + '\\' + sub_trial + '\\' + emo, 'r')
                emo_num = f.read()[3:4]  # e.g '   4.0000000e+00'
                emo_type = emo_dict[int(emo_num)]
                img_label.append([sub,sub_trial,emo_num,emo_type])
    
    df = pd.DataFrame(img_label, columns=['sub','sub_trial','emo_num','emo_type'])
    
    len(img_label) #327
    label_distribution = df.groupby('emo_type').size() #n = 45,18,59,25,69,28,83
    
    return img_label


# 1.2 Split into training and testing set
def split_labelled_img_from_CK_plus(img_label):
    # Make folders for label
    os.makedirs('test')
    for i in range(1,8):
        os.makedirs('train\\'+emo_dict[i])

    # train:test = 80:20
    random.seed(2)
    test_set = random.sample(img_label,65) 
    train_set = [x for x in img_label if x not in test_set]
    
    return test_set,train_set


# 1.3 Copy images from original dataset (source) to the train/test folders
def copy_img_traintest_from_CK_plus (test_set,train_set,ROOT):
    TRAIN_PATH = ROOT + 'train\\'
    TEST_PATH = ROOT + 'test\\'
    IMAGE_SOURCE_PATH = ROOT + '..\\' +'extended-cohn-kanade-images\\cohn-kanade-images\\'
    
    # Copy to train set 
    for i in range(0,len(train_set)):
        
        # Get path to the image folder
        data = train_set[i] #['sub','sub_trial','emo_num','emo_type']
        data_PATH = os.path.join(IMAGE_SOURCE_PATH,data[0],data[1])
        
        # Get path to the last image in folder
        last_image = os.listdir(data_PATH)[-1]
        last_image_PATH = os.path.join(data_PATH,last_image)
        src = last_image_PATH
        
        # Get path to label folder
        dst = os.path.join(TRAIN_PATH,data[-1])
        
        copy2(src,dst)
        
    # Copy to test set 
    for i in range(0,len(test_set)):
        data = test_set[i] #['sub','sub_trial','emo_num','emo_type']
        data_PATH = os.path.join(IMAGE_SOURCE_PATH,data[0],data[1])
        
        last_image = os.listdir(data_PATH)[-1]
        last_image_PATH = os.path.join(data_PATH,last_image)
        src = last_image_PATH
        
        # Get path to label folder AND append label as prefix to filename
        dst = TEST_PATH + data[-1] + '_' + last_image
        
        copy2(src,dst)

# 1.4 Convert to greyscale, convert png to jpg
def run_make_grey_and_jpg(ROOT):
    TRAIN_PATH = ROOT + 'train\\'
    TEST_PATH = ROOT + 'test\\'
    
    train_list = glob(TRAIN_PATH + '\\*\*')
    test_list = glob(TEST_PATH + '\\*')

    for data_list in [train_list,test_list]:
        for img in data_list:
            
            new_img = cv2.imread(img,0)
            filename = img[0:-3] + 'jpg'
            cv2.imwrite(filename,new_img)            
            if img.endswith('png'):
                os.remove(img)

# 1.5 Data augmentation for Contempt, Fear, Sadness which has 
# ~ 20 training images only. 
def add_flipped_images (ROOT, emotion):
    EMO_PATH = os.path.join(ROOT,'train_crop',emotion) + '\\'
    for img in os.listdir(EMO_PATH):
        
        bef_img = cv2.imread(EMO_PATH + img, 0)
        aft_img = bef_img[:,::-1]

        img = img[:9] + 'flip' + img[13:]
        cv2.imwrite(EMO_PATH + img, aft_img)
    
        
if __name__ == "__main__":
    os.chdir(ROOT)
    
    img_label = extract_label_from_CK_plus()
    test_set,train_set = split_labelled_img_from_CK_plus(img_label)
    copy_img_traintest_from_CK_plus (test_set,train_set,ROOT)
   
    run_make_grey_and_jpg(ROOT)
    
    from crop_face import *
    run_crop_face(ROOT)

    add_flipped_images (ROOT,'Contempt')
    add_flipped_images (ROOT,'Fear')
    add_flipped_images (ROOT,'Sadness')
    
    from threshold_face import *
    run_threshold_face (ROOT)
    
    from subtract_neutral_face import *
    run_subtract_neutral_face (ROOT)
