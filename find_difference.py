import os
from data_extraction_preprocessing import * #for the emo_dict map

# Find difference in images between author(Pao)'s dataset and ours
pao_TRAINING_PATH = 'C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\code_pao\\EE368 Final Project MATLAB JPao'
our_TRAINING_PATH = ROOT + 'train'

count_added_in=0
count_dropped=0
for i in range(1,8):    #'0' is Neutral emotion which is not required
    print (emo_dict[i])
    emo_count_from_pao = 0
    emo_count_from_us = 0
    
    #Pao named some emotions differently from Cohn-Kanade dataset
    if i == 1:
        pao_list = os.listdir(os.path.join(pao_TRAINING_PATH,'Angry'))        
    elif i == 6:
        pao_list = os.listdir(os.path.join(pao_TRAINING_PATH,'Sad')) 
    else:
        pao_list = os.listdir(os.path.join(pao_TRAINING_PATH,emo_dict[i]))
        
    our_list = [x[0:17] for x in os.listdir(os.path.join(our_TRAINING_PATH,emo_dict[i]))]
    pao_list = [x[0:17] for x in pao_list]
    
    for img in pao_list:
        if img not in our_list:
            print (img)
            count_added_in +=1
            emo_count_from_pao +=1
    print ('Number of image added in for {}: {}\n'.format(emo_dict[i], emo_count_from_pao))           
    
    for img in our_list:
        if img not in pao_list:
            print (img)
            count_dropped +=1
            emo_count_from_us +=1
    print ('Number of image dropped from {}: {}\n'.format(emo_dict[i], emo_count_from_us))
print ('Total number of image added in: ' + str(count_added_in))
print ('Total number of image dropped out: ' + str(count_dropped))
