import time
import os
import subprocess
import shlex
import pandas as pd

# Path to root
ROOT = "C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\"
os.chdir(ROOT)

from tensorflow_for_poets_2.scripts.label_image import *

# CONFIG FOR RETRAINING
# image_dir = directory of training data
# output_graph = location to save graph
# output_labels = labels in training data
# architecture = model to retrain on
# Change the variable after the flag. 
#   E.g ' --how_many_training_steps 200'. Just change 200.

image_dir = ' --image_dir train_crop'
output_graph = ' --output_graph tensorflow_for_poets_2/tf_files/output_graph_m4000_crop.pb'
output_labels = ' --output_labels tensorflow_for_poets_2/tf_files/output_labels.txt'
how_many_training_steps = ' --how_many_training_steps 4000'
architecture = ' --architecture mobilenet_1.0_224' #either mobilenet_1.0_224 or inception_v3

# CONFIG FOR LABELLING or PREDICTION
# label_image.py for prediction is small and the default ouput is not 
# suitable for integration with our video demo, video.py. Hence the code is 
# copied and modified here, which also explains why the variable inputs for 
# retraining and labelling are different

TEST_FOLDER_PATH = ROOT + 'test_crop\\'
model_file = 'tensorflow_for_poets_2\\tf_files\\output_graph_m4000_crop.pb' 
label_file = 'tensorflow_for_poets_2\\tf_files\\output_labels.txt'
input_height = 224 #299 for inception, 224 for mobilenet 
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input" #"Mul" for inception, "input" for mobilenet
output_layer = "final_result"


# BEGIN RETRAINING CNN MODEL
BASE_CMD = 'python -m tensorflow_for_poets_2.scripts.retrain'
ARGS_CMD = ''.join((image_dir,
                    output_graph,
                    output_labels,
                    how_many_training_steps,
                    architecture
                    ))
CMD = BASE_CMD + ARGS_CMD

# use this line to test a small py script on the subprocess.Popen module
# args = shlex.split('python -m test --input xyz')

args = shlex.split(CMD)
t_start=time.time()
print(subprocess.Popen(args,stdout=subprocess.PIPE).stdout.read().decode())
t_end_training=time.time()

# LABELLING TEST DATA
def run_label(file_name, PREDICTION_AS_LIST=0): # Give prediction for one image
  graph = load_graph(model_file)
  t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()
  results = np.squeeze(results)

  top_k = results.argsort()[-3:][::-1]
  labels = load_labels(label_file)

  # It is desirable to save prediction as list, which allows concatenation into dataframe for export later
  # It is also desirable to just output the top label by default
  if PREDICTION_AS_LIST:
      #print('Evaluation time: {:.3f}s'.format(end-start))
      output_list = []
      #template = "{} (score={:0.5f})"
      for i in top_k:
          temp = [labels[i],results[i]]
          output_list += temp
    
      return output_list  
  
  else:
      return labels[top_k[0]]

def prediction2csv (TEST_FOLDER_PATH): # Give prediction for one folder of images, measure accuracy
    listfile=os.listdir(TEST_FOLDER_PATH)
    outcome = []
    for file in listfile:
        if file.endswith('jpg'):
            file_label = file.split('_')[0].lower()
            temp = [file] + [file_label] + run_label(TEST_FOLDER_PATH+file, PREDICTION_AS_LIST=True)
            #print (file)
            #runlabel(TEST_PATH+file)
            #print ()
            outcome.append(temp)

    df_o = pd.DataFrame(outcome, columns=['File','Label','Pred1','Pred1_score','p2','p2score','p3','p3score'])

    # Print accuracy in console 
    count=0
    for i in range(0,len(df_o)):
        label = df_o.iloc[i,1]
        pred = df_o.iloc[i,2]
        
        if label == pred:
            count=count+1
            
    print ('Train:' + image_dir)
    print (os.path.split(model_file)[-1])
    print (how_many_training_steps)
    print ('Accuracy = {}/{} = {}%\n'.format(count,len(df_o),count/len(df_o)*100))
    print (pd.crosstab(df_o['Label'], df_o['Pred1'],margins=1))
    
    pd.DataFrame.to_csv(df_o,'_'.join((image_dir.split()[1],
                                 how_many_training_steps.split()[1],
                                 architecture.split()[1],
                                 ".csv")))

prediction2csv (TEST_FOLDER_PATH)
t_end_labelling=time.time()

print ('Training time: ' + str(t_end_training-t_start) + 'sec')
print ('Labelling time: ' + str(t_end_labelling-t_end_training) + 'sec')

# incep, _substract (threshold=10), 500s, 15%
# incep, _substract (diff), 500s, 15%
# incep, _substract (ath), 500s, 16%
# incep, _threshold, 500s, 27%
# incep, jc, 500s, 73%
# incep, jc + random brightness, 500s, 70%
# mobilenet, author's training set, 500, 80%

# mobilenet, jc [60] 7%
# mobilenet, jc [125] 12%
# mobilenet, jc [250] 9%
# mobilenet, jc [500] 9%
# mobilenet, jc [1000] 9%

# mobilenet, th [125] 13%
# mobilenet, th [250] 13%
# mobilenet, th [500] 12%
# mobilenet, th [1000] 12%

# mobilenet, subt [125], 29%
# mobilenet, subt [250], 80%
# mobilenet, subt [500], 78%
# mobilenet, subt [1000], 81%

# mobilenet, auth [125], 76%
# mobilenet, auth [250], 83%
# mobilenet, auth [500], 78%
# mobilenet, auth [1000], 80%

# inception, auth [125], 60%
# inception, auth [250], 63%
# inception, auth [500], 63%
# inception, auth [1000], 66%

# inception, subt [125], 4%
# inception, subt [250], 7%
# inception, subt [500], 9%
# inception, subt [1000], 15%

# inception, th [125], 4%
# inception, th [250], 7%
# inception, th [500], 7%
# inception, th [1000],7%

# inception, jc [125], 66%
# inception, jc [250], 69%
# inception, jc [500], 72%
# inception, jc [1000], 72%

################
### RESULTS ####
################



#Train: --image_dir train_threshold
#output_graph_m_crop.pb
#Accuracy = 7/65 = 10.76923076923077%
#
#Pred1     anger  contempt  disgust  fear  sadness  surprise  All
#Label                                                           
#anger         1         3        0     0        1         0    5
#contempt      0         3        0     0        0         0    3
#disgust       0        12        1     0        0         0   13
#fear          0         3        0     0        0         0    3
#happy         0        14        0     0        0         0   14
#sadness       0         6        0     1        0         0    7
#surprise      0        18        0     0        0         2   20
#All           1        59        1     1        1         2   65

#inception, jc, 70%
#Train: --image_dir train_crop
#output_graph_i_crop.pb
#Accuracy = 46/65 = 70.76923076923077%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         5         0        0     0      0        0         0    5
#contempt      1         2        0     0      0        0         0    3
#disgust       2         3        7     1      0        0         0   13
#fear          0         1        0     2      0        0         0    3
#happy         0         0        1     0     13        0         0   14
#sadness       5         1        0     1      0        0         0    7
#surprise      0         1        0     1      0        1        17   20
#All          13         8        8     5     13        1        17   65

#Train: --image_dir train_crop
#output_graph_i_crop.pb
# --how_many_training_steps 125,250
#Accuracy = 39/65 = 60.0%
#
#Pred1     anger  contempt  disgust  happy  surprise  All
#Label                                                   
#anger         4         1        0      0         0    5
#contempt      2         0        0      1         0    3
#disgust       3         2        5      3         0   13
#fear          0         0        1      1         1    3
#happy         0         0        1     13         0   14
#sadness       5         1        0      0         1    7
#surprise      1         1        0      1        17   20
#All          15         5        7     19        19   65

#Train: --image_dir train_crop
#output_graph_i_crop.pb
# --how_many_training_steps 500,1000
#Accuracy = 47/65 = 72.3076923076923%
#
#Pred1     anger  contempt  disgust  fear  happy  surprise  All
#Label                                                         
#anger         5         0        0     0      0         0    5
#contempt      1         2        0     0      0         0    3
#disgust       1         3        8     1      0         0   13
#fear          0         1        0     2      0         0    3
#happy         0         0        1     0     13         0   14
#sadness       5         1        0     0      0         1    7
#surprise      0         2        0     1      0        17   20
#All          12         9        9     4     13        18   65

#Train: --image_dir train_threshold
#output_graph_i_threshold.pb
# --how_many_training_steps 250,500,1000
#Accuracy = 5/65 = 7.6923076923076925%
#
#Pred1     fear  sadness  All
#Label                       
#anger        1        4    5
#contempt     1        2    3
#disgust      6        7   13
#fear         1        2    3
#happy       12        2   14
#sadness      3        4    7
#surprise    14        6   20
#All         38       27   65

#Train: --image_dir train_subtract
#output_graph_i_subtract.pb
# --how_many_training_steps 250,500,1000
#Accuracy = 7/65 = 10.76923076923077%
#
#Pred1     anger  contempt  disgust  sadness  All
#Label                                           
#anger         0         4        0        1    5
#contempt      0         3        0        0    3
#disgust       1         9        3        0   13
#fear          0         3        0        0    3
#happy         1        12        1        0   14
#sadness       0         6        0        1    7
#surprise      0        19        0        1   20
#All           2        56        4        3   65

#Train: --image_dir train_subtract
#output_graph_m500_subtract.pb
# --how_many_training_steps 500
#Accuracy = 52/65 = 80.0%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         2         0        2     1      0        0         0    5
#contempt      0         1        0     1      0        1         0    3
#disgust       0         0       12     0      0        1         0   13
#fear          0         0        0     3      0        0         0    3
#happy         0         0        0     0     14        0         0   14
#sadness       2         0        0     0      0        4         1    7
#surprise      1         1        0     2      0        0        16   20
#All           5         2       14     7     14        6        17   65

#Train: --image_dir train_subtract
#output_graph_m250_subtract.pb
# --how_many_training_steps 250
#Accuracy = 52/65 = 80.0%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         2         0        1     0      1        1         0    5
#contempt      0         1        0     1      0        1         0    3
#disgust       0         0       11     0      1        1         0   13
#fear          0         0        0     3      0        0         0    3
#happy         0         0        0     0     14        0         0   14
#sadness       2         0        0     0      0        5         0    7
#surprise      0         1        0     2      0        1        16   20
#All           4         2       12     6     16        9        16   65

#Train: --image_dir train_subtract
#output_graph_m1000_subtract.pb
# --how_many_training_steps 1000
#Accuracy = 47/65 = 72.3076923076923%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         1         0        1     1      1        1         0    5
#contempt      0         1        0     1      0        1         0    3
#disgust       0         0       10     1      1        1         0   13
#fear          0         0        0     3      0        0         0    3
#happy         0         0        0     0     14        0         0   14
#sadness       3         0        0     0      0        3         1    7
#surprise      1         1        0     3      0        0        15   20
#All           5         2       11     9     16        6        16   65

#Train: --image_dir train_crop
#output_graph_m1000_crop.pb
# --how_many_training_steps 1000
#Accuracy = 6/65 = 9.230769230769232%
#
#Pred1     contempt  disgust  sadness  surprise  All
#Label                                              
#anger            3        0        2         0    5
#contempt         3        0        0         0    3
#disgust         12        1        0         0   13
#fear             3        0        0         0    3
#happy           14        0        0         0   14
#sadness          7        0        0         0    7
#surprise        18        0        0         2   20
#All             60        1        2         2   65

#Train: --image_dir train_crop
#output_graph_m500_crop.pb
# --how_many_training_steps 500
#Accuracy = 6/65 = 9.230769230769232%
#
#Pred1     contempt  disgust  fear  sadness  surprise  All
#Label                                                    
#anger            3        0     0        2         0    5
#contempt         3        0     0        0         0    3
#disgust         12        1     0        0         0   13
#fear             3        0     0        0         0    3
#happy           14        0     0        0         0   14
#sadness          4        0     3        0         0    7
#surprise        18        0     0        0         2   20
#All             57        1     3        2         2   65

#Train: --image_dir train_crop
#output_graph_m250_crop.pb
# --how_many_training_steps 250
#Accuracy = 6/65 = 9.230769230769232%
#
#Pred1     contempt  disgust  fear  sadness  surprise  All
#Label                                                    
#anger            3        0     0        2         0    5
#contempt         3        0     0        0         0    3
#disgust         12        1     0        0         0   13
#fear             3        0     0        0         0    3
#happy           13        1     0        0         0   14
#sadness          4        0     3        0         0    7
#surprise        18        0     0        0         2   20
#All             56        2     3        2         2   65

#Train: --image_dir train_crop
#output_graph_m125_crop.pb
# --how_many_training_steps 125
#Accuracy = 8/65 = 12.307692307692308%
#
#Pred1     anger  contempt  disgust  fear  sadness  surprise  All
#Label                                                           
#anger         0         2        0     1        2         0    5
#contempt      0         3        0     0        0         0    3
#disgust       0        10        3     0        0         0   13
#fear          0         3        0     0        0         0    3
#happy         0         7        2     5        0         0   14
#sadness       1         2        1     3        0         0    7
#surprise      0        15        0     3        0         2   20
#All           1        42        6    12        2         2   65

#Train: --image_dir train_crop
#output_graph_m60_crop.pb
# --how_many_training_steps 60
#Accuracy = 5/65 = 7.6923076923076925%
#
#Pred1     contempt  disgust  fear  sadness  All
#Label                                          
#anger            3        0     1        1    5
#contempt         3        0     0        0    3
#disgust         11        2     0        0   13
#fear             3        0     0        0    3
#happy            7        2     5        0   14
#sadness          2        1     4        0    7
#surprise        13        0     7        0   20
#All             42        5    17        1   65

#Train: --image_dir train_threshold
#output_graph_m125_threshold.pb
# --how_many_training_steps 125
#Accuracy = 9/65 = 13.846153846153847%
#
#Pred1     fear  sadness  All
#Label                       
#anger        1        4    5
#contempt     1        2    3
#disgust      4        9   13
#fear         3        0    3
#happy       10        4   14
#sadness      1        6    7
#surprise    11        9   20
#All         31       34   65

#Train: --image_dir train_threshold
#output_graph_m250_threshold.pb
# --how_many_training_steps 250
#Accuracy = 9/65 = 13.846153846153847%
#
#Pred1     fear  sadness  All
#Label                       
#anger        1        4    5
#contempt     1        2    3
#disgust      4        9   13
#fear         3        0    3
#happy        9        5   14
#sadness      1        6    7
#surprise    11        9   20
#All         30       35   65

#Train: --image_dir train_threshold
#output_graph_m500_threshold.pb
# --how_many_training_steps 500
#Accuracy = 8/65 = 12.307692307692308%
#
#Pred1     fear  sadness  All
#Label                       
#anger        1        4    5
#contempt     1        2    3
#disgust      4        9   13
#fear         3        0    3
#happy       12        2   14
#sadness      2        5    7
#surprise    10       10   20
#All         33       32   65

#Train: --image_dir train_threshold
#output_graph_m1000_threshold.pb
# --how_many_training_steps 1000
#Accuracy = 8/65 = 12.307692307692308%
#
#Pred1     fear  sadness  All
#Label                       
#anger        1        4    5
#contempt     1        2    3
#disgust      4        9   13
#fear         3        0    3
#happy       12        2   14
#sadness      2        5    7
#surprise    11        9   20
#All         34       31   65

#Train: --image_dir train_subtract
#output_graph_m125_subtract.pb
# --how_many_training_steps 125
#Accuracy = 49/65 = 75.38461538461539%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         2         0        1     0      1        1         0    5
#contempt      0         1        0     1      0        1         0    3
#disgust       0         0       12     0      0        1         0   13
#fear          0         0        0     3      0        0         0    3
#happy         0         0        0     0     14        0         0   14
#sadness       3         1        0     0      0        2         1    7
#surprise      0         2        0     1      0        2        15   20
#All           5         4       13     5     15        7        16   65

#Train: --image_dir train_subtract
#output_graph_m250_subtract.pb
# --how_many_training_steps 250
#Accuracy = 52/65 = 80.0%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         2         0        2     1      0        0         0    5
#contempt      0         1        0     1      0        1         0    3
#disgust       0         0       11     1      0        1         0   13
#fear          0         0        0     3      0        0         0    3
#happy         0         0        0     0     14        0         0   14
#sadness       1         0        0     0      0        5         1    7
#surprise      0         1        0     2      0        1        16   20
#All           3         2       13     8     14        8        17   65

#Train: --image_dir train_subtract
#output_graph_m500_subtract.pb
# --how_many_training_steps 500
#Accuracy = 51/65 = 78.46153846153847%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         2         0        2     1      0        0         0    5
#contempt      0         1        0     1      0        1         0    3
#disgust       0         0       11     0      1        1         0   13
#fear          0         0        0     3      0        0         0    3
#happy         0         0        0     0     14        0         0   14
#sadness       2         0        0     0      0        4         1    7
#surprise      1         1        0     2      0        0        16   20
#All           5         2       13     7     15        6        17   65

#Train: --image_dir train_subtract
#output_graph_m1000_subtract.pb
# --how_many_training_steps 1000
#Accuracy = 53/65 = 81.53846153846153%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         2         0        1     1      1        0         0    5
#contempt      0         1        0     1      0        1         0    3
#disgust       0         0       12     0      0        1         0   13
#fear          0         0        0     3      0        0         0    3
#happy         0         0        0     0     14        0         0   14
#sadness       2         0        0     0      0        5         0    7
#surprise      1         1        0     2      0        0        16   20
#All           5         2       13     7     15        7        16   65

#Train: --image_dir train_author
#output_graph_m125_auth.pb
# --how_many_training_steps 125
#Accuracy = 50/65 = 76.92307692307693%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         1         2        0     0      0        2         0    5
#contempt      0         1        0     1      1        0         0    3
#disgust       0         1        9     0      3        0         0   13
#fear          0         0        0     2      1        0         0    3
#happy         0         0        0     1     13        0         0   14
#sadness       0         0        1     1      0        5         0    7
#surprise      0         0        0     0      1        0        19   20
#All           1         4       10     5     19        7        19   65

#Train: --image_dir train_author
#output_graph_m250_auth.pb
# --how_many_training_steps 250
#Accuracy = 54/65 = 83.07692307692308%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         1         2        0     0      0        2         0    5
#contempt      0         2        0     1      0        0         0    3
#disgust       0         1       10     0      2        0         0   13
#fear          0         0        0     3      0        0         0    3
#happy         0         0        0     1     13        0         0   14
#sadness       0         0        1     0      0        6         0    7
#surprise      0         0        0     0      1        0        19   20
#All           1         5       11     5     16        8        19   65

#Train: --image_dir train_author
#output_graph_m500_auth.pb
# --how_many_training_steps 500
#Accuracy = 51/65 = 78.46153846153847%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         1         2        0     0      0        2         0    5
#contempt      0         2        0     1      0        0         0    3
#disgust       0         2        9     0      2        0         0   13
#fear          0         0        0     3      0        0         0    3
#happy         0         0        0     1     12        0         1   14
#sadness       0         0        1     1      0        5         0    7
#surprise      0         0        0     0      1        0        19   20
#All           1         6       10     6     15        7        20   65

#Train: --image_dir train_author
#output_graph_m1000_auth.pb
# --how_many_training_steps 1000
#Accuracy = 52/65 = 80.0%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         1         2        0     0      0        2         0    5
#contempt      0         1        0     1      1        0         0    3
#disgust       0         1       10     0      2        0         0   13
#fear          0         0        0     3      0        0         0    3
#happy         0         0        0     1     12        0         1   14
#sadness       0         0        1     0      0        6         0    7
#surprise      0         0        0     0      1        0        19   20
#All           1         4       11     5     16        8        20   65

#Train: --image_dir train_author
#output_graph_i125_auth.pb
# --how_many_training_steps 125
#Accuracy = 39/65 = 60.0%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         1         4        0     0      0        0         0    5
#contempt      0         0        0     0      0        3         0    3
#disgust       1         2        4     0      3        3         0   13
#fear          0         0        0     1      1        0         1    3
#happy         0         1        0     0     13        0         0   14
#sadness       0         3        0     0      0        2         2    7
#surprise      0         2        0     0      0        0        18   20
#All           2        12        4     1     17        8        21   65

#Train: --image_dir train_author
#output_graph_i250_auth.pb
# --how_many_training_steps 250
#Accuracy = 41/65 = 63.07692307692307%
#
#Pred1     anger  contempt  disgust  happy  sadness  surprise  All
#Label                                                            
#anger         1         4        0      0        0         0    5
#contempt      0         0        0      0        3         0    3
#disgust       0         1        5      4        3         0   13
#fear          0         0        1      1        0         1    3
#happy         0         1        0     13        0         0   14
#sadness       0         2        0      0        3         2    7
#surprise      0         1        0      0        0        19   20
#All           1         9        6     18        9        22   65

#Train: --image_dir train_author
#output_graph_i500_auth.pb
# --how_many_training_steps 500
#Accuracy = 41/65 = 63.07692307692307%
#
#Pred1     anger  contempt  disgust  happy  sadness  surprise  All
#Label                                                            
#anger         1         4        0      0        0         0    5
#contempt      0         0        0      0        3         0    3
#disgust       1         2        4      3        3         0   13
#fear          0         0        1      1        0         1    3
#happy         0         1        0     13        0         0   14
#sadness       0         1        0      0        4         2    7
#surprise      0         1        0      0        0        19   20
#All           2         9        5     17       10        22   65

#Train: --image_dir train_author
#output_graph_i1000_auth.pb
# --how_many_training_steps 1000
#Accuracy = 43/65 = 66.15384615384615%
#
#Pred1     anger  contempt  disgust  happy  sadness  surprise  All
#Label                                                            
#anger         1         3        0      0        1         0    5
#contempt      0         0        0      0        3         0    3
#disgust       0         1        5      2        4         1   13
#fear          0         0        1      0        0         2    3
#happy         0         1        0     13        0         0   14
#sadness       0         0        0      0        5         2    7
#surprise      0         1        0      0        0        19   20
#All           1         6        6     15       13        24   65

#Train: --image_dir train_subtract
#output_graph_i125_subtract.pb
# --how_many_training_steps 125
#Accuracy = 3/65 = 4.615384615384616%
#
#Pred1     contempt  sadness  All
#Label                           
#anger            5        0    5
#contempt         3        0    3
#disgust         12        1   13
#fear             3        0    3
#happy           14        0   14
#sadness          7        0    7
#surprise        20        0   20
#All             64        1   65

#Train: --image_dir train_subtract
#output_graph_i250_subtract.pb
# --how_many_training_steps 250
#Accuracy = 5/65 = 7.6923076923076925%
#
#Pred1     contempt  disgust  sadness  All
#Label                                    
#anger            5        0        0    5
#contempt         3        0        0    3
#disgust         11        1        1   13
#fear             3        0        0    3
#happy           14        0        0   14
#sadness          6        0        1    7
#surprise        19        0        1   20
#All             61        1        3   65

#Train: --image_dir train_subtract
#output_graph_i500_subtract.pb
# --how_many_training_steps 500
#Accuracy = 6/65 = 9.230769230769232%
#
#Pred1     anger  contempt  disgust  sadness  All
#Label                                           
#anger         0         4        0        1    5
#contempt      0         3        0        0    3
#disgust       1         9        3        0   13
#fear          0         3        0        0    3
#happy         1        11        2        0   14
#sadness       1         6        0        0    7
#surprise      0        19        0        1   20
#All           3        55        5        2   65
#Training time: 106.44545340538025sec
#Labelling time: 114.89184260368347sec

#Train: --image_dir train_subtract
#output_graph_i1000_subtract.pb
# --how_many_training_steps 1000
#Accuracy = 10/65 = 15.384615384615385%
#
#Pred1     anger  contempt  disgust  fear  sadness  surprise  All
#Label                                                           
#anger         1         3        0     0        1         0    5
#contempt      0         3        0     0        0         0    3
#disgust       1         9        3     0        0         0   13
#fear          0         3        0     0        0         0    3
#happy         1        10        3     0        0         0   14
#sadness       1         5        0     1        0         0    7
#surprise      0        15        0     1        1         3   20
#All           4        48        6     2        2         3   65

#Train: --image_dir train_threshold
#output_graph_i125_threshold.pb
# --how_many_training_steps 125
#Accuracy = 3/65 = 4.615384615384616%
#
#Pred1     contempt  sadness  All
#Label                           
#anger            5        0    5
#contempt         3        0    3
#disgust         12        1   13
#fear             3        0    3
#happy           14        0   14
#sadness          7        0    7
#surprise        20        0   20
#All             64        1   65
#Training time: 43.28221130371094sec
#Labelling time: 133.0275559425354sec

#Train: --image_dir train_threshold
#output_graph_i250_threshold.pb
# --how_many_training_steps 250
#Accuracy = 5/65 = 7.6923076923076925%
#
#Pred1     fear  sadness  All
#Label                       
#anger        0        5    5
#contempt     0        3    3
#disgust      4        9   13
#fear         1        2    3
#happy       12        2   14
#sadness      3        4    7
#surprise     8       12   20
#All         28       37   65

#Train: --image_dir train_threshold
#output_graph_i500_threshold.pb
# --how_many_training_steps 500
#Accuracy = 5/65 = 7.6923076923076925%
#
#Pred1     fear  sadness  All
#Label                       
#anger        1        4    5
#contempt     1        2    3
#disgust      6        7   13
#fear         1        2    3
#happy       12        2   14
#sadness      3        4    7
#surprise    14        6   20
#All         38       27   65

#Train: --image_dir train_threshold
#output_graph_i1000_threshold.pb
# --how_many_training_steps 1000
#Accuracy = 5/65 = 7.6923076923076925%
#
#Pred1     fear  sadness  All
#Label                       
#anger        1        4    5
#contempt     0        3    3
#disgust      3       10   13
#fear         1        2    3
#happy       12        2   14
#sadness      3        4    7
#surprise    12        8   20
#All         32       33   65

#Train: --image_dir train_crop
#output_graph_i125_crop.pb
# --how_many_training_steps 125
#Accuracy = 43/65 = 66.15384615384615%
#
#Pred1     anger  contempt  disgust  happy  surprise  All
#Label                                                   
#anger         5         0        0      0         0    5
#contempt      2         0        0      1         0    3
#disgust       4         1        6      2         0   13
#fear          0         0        1      1         1    3
#happy         0         0        1     13         0   14
#sadness       5         1        0      0         1    7
#surprise      1         0        0      0        19   20
#All          17         2        8     17        21   65
#Training time: 45.20264029502869sec
#Labelling time: 179.421240568161sec

#Train: --image_dir train_crop
#output_graph_i250_crop.pb
# --how_many_training_steps 250
#Accuracy = 45/65 = 69.23076923076923%
#
#Pred1     anger  contempt  disgust  fear  happy  surprise  All
#Label                                                         
#anger         5         0        0     0      0         0    5
#contempt      1         2        0     0      0         0    3
#disgust       3         2        7     0      1         0   13
#fear          0         0        1     0      1         1    3
#happy         0         0        1     0     13         0   14
#sadness       5         1        0     0      0         1    7
#surprise      1         0        0     1      0        18   20
#All          15         5        9     1     15        20   65

#Train: --image_dir train_crop
#output_graph_i500_crop.pb
# --how_many_training_steps 500
#Accuracy = 47/65 = 72.3076923076923%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         5         0        0     0      0        0         0    5
#contempt      1         2        0     0      0        0         0    3
#disgust       1         3        8     0      1        0         0   13
#fear          0         1        0     2      0        0         0    3
#happy         0         0        1     0     13        0         0   14
#sadness       4         1        1     0      0        0         1    7
#surprise      0         1        0     1      0        1        17   20
#All          11         8       10     3     14        1        18   65

#Train: --image_dir train_crop
#output_graph_i1000_crop.pb
# --how_many_training_steps 1000
#Accuracy = 47/65 = 72.3076923076923%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         5         0        0     0      0        0         0    5
#contempt      0         3        0     0      0        0         0    3
#disgust       1         4        8     0      0        0         0   13
#fear          0         1        0     2      0        0         0    3
#happy         0         0        1     0     13        0         0   14
#sadness       4         1        1     0      0        0         1    7
#surprise      0         2        0     1      0        1        16   20
#All          10        11       10     3     13        1        17   65

#Train: --image_dir train_subtract
#output_graph_m250_subtract_time.pb
# --how_many_training_steps 250
#Accuracy = 53/65 = 81.53846153846153%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         2         0        1     0      1        1         0    5
#contempt      0         1        0     1      0        1         0    3
#disgust       0         0       12     0      0        1         0   13
#fear          0         0        0     3      0        0         0    3
#happy         0         0        0     0     14        0         0   14
#sadness       2         0        0     0      0        4         1    7
#surprise      1         1        0     1      0        0        17   20
#All           5         2       13     5     15        7        18   65
#Training time: 30.607271194458008sec
#Labelling time: 64.45180249214172sec
#
#Train: --image_dir train_subtract
#output_graph_i250_subtract_time.pb
# --how_many_training_steps 250
#Accuracy = 4/65 = 6.153846153846154%
#
#Pred1     contempt  disgust  sadness  All
#Label                                    
#anger            5        0        0    5
#contempt         3        0        0    3
#disgust         11        1        1   13
#fear             3        0        0    3
#happy           14        0        0   14
#sadness          7        0        0    7
#surprise        20        0        0   20
#All             63        1        1   65
#Training time: 57.33262586593628sec
#Labelling time: 98.38115859031677sec

#Train: --image_dir train_crop
#output_graph_i4000_crop.pb
# --how_many_training_steps 4000
#Accuracy = 46/65 = 70.76923076923077%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         5         0        0     0      0        0         0    5
#contempt      1         2        0     0      0        0         0    3
#disgust       1         3        8     1      0        0         0   13
#fear          0         1        0     2      0        0         0    3
#happy         0         0        1     0     13        0         0   14
#sadness       5         1        0     1      0        0         0    7
#surprise      0         2        0     1      0        1        16   20
#All          12         9        9     5     13        1        16   65
#Training time: 853.1584942340851sec
#Labelling time: 248.04022479057312sec

#Train: --image_dir train_author
#output_graph_i4000_author.pb
# --how_many_training_steps 4000
#Accuracy = 43/65 = 66.15384615384615%
#
#Pred1     anger  contempt  disgust  happy  sadness  surprise  All
#Label                                                            
#anger         1         0        0      0        4         0    5
#contempt      0         0        0      0        3         0    3
#disgust       0         1        4      1        6         1   13
#fear          0         0        0      0        1         2    3
#happy         0         1        0     13        0         0   14
#sadness       0         0        0      0        6         1    7
#surprise      0         1        0      0        0        19   20
#All           1         3        4     14       20        23   65
#Training time: 609.0174136161804sec
#Labelling time: 101.63457560539246sec

#Train: --image_dir train_subtract
#output_graph_m4000_subtract.pb
# --how_many_training_steps 4000
#Accuracy = 50/65 = 76.92307692307693%
#
#Pred1     anger  contempt  disgust  fear  happy  sadness  surprise  All
#Label                                                                  
#anger         1         0        1     1      1        1         0    5
#contempt      0         1        0     1      0        1         0    3
#disgust       0         0       11     0      1        1         0   13
#fear          0         0        0     3      0        0         0    3
#happy         0         0        0     0     14        0         0   14
#sadness       3         0        0     0      0        4         0    7
#surprise      1         1        0     2      0        0        16   20
#All           5         2       12     7     16        7        16   65
#Training time: 392.68836855888367sec
#Labelling time: 69.84935975074768sec

