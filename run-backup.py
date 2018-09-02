# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 22:54:31 2018

@author: peanut
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 22:14:32 2018

@author: peanut
"""


import os
import subprocess
import shlex

import pandas as pd

PATH = "C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\"
os.chdir(PATH)

#from tensorflow_for_poets_2.scripts.retrain import *
#retraining config
image_dir = ' --image_dir train_crop'
output_graph = ' --output_graph tensorflow_for_poets_2/tf_files/output_graph_icpt3.pb'
output_labels = ' --output_labels tensorflow_for_poets_2//tf_files//output_labels.txt'
how_many_training_steps = ' --how_many_training_steps 500'
architecture = ' --architecture inception_v3'

BASE_CMD = 'python -m tensorflow_for_poets_2.scripts.retrain'
ARGS_CMD = image_dir + output_graph + output_labels + how_many_training_steps +architecture
CMD = BASE_CMD + ARGS_CMD

args = shlex.split(CMD)
#arg = shlex.split('python -m tensorflow_for_poets_2.test --input xyz')

print(subprocess.Popen(args,stdout=subprocess.PIPE).stdout.read().decode())

parser = argparse.ArgumentParser()
parser.add_argument(
        '--image_dir',
        type=str,
        default='train_author\\', #train\\
        help='Path to folders of labeled images.'
        )
parser.add_argument(
        '--output_graph',
        default = 'tensorflow_for_poets_2\\tf_files\\output_graph_incep_author.pb',
        help='Where to save the trained graph.'
        )
parser.add_argument(
      '--output_labels',
      type=str,
      default='tensorflow_for_poets_2\\tf_files\\output_labels.txt',
      help='Where to save the trained graph\'s labels.'
      )
parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=500,
      help='How many training steps to run before ending.'
      )


parser.add_argument(
        '--intermediate_output_graphs_dir',
        default='/tmp/intermediate_graph/'
        )
parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0
  ) 
parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs' ,
      help='Where to save summary logs for TensorBoard.'
  )
  
parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )

parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  
parser.add_argument(
      '--model_dir',
      type=str,
      default='tensorflow_for_poets_2\\tf_files\\models',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
parser.add_argument(
      '--flip_left_right',
      default=False,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\.
      """
  )
  
parser.add_argument(
      '--architecture',
      type=str,
      default='inception_v3',
      help="""\
      Which model architecture to use. 'inception_v3' is the most accurate, but
      also the slowest. For faster or smaller models, chose a MobileNet with the
      form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
      'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
      pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
      less accurate, but smaller and faster network that's 920 KB on disk and
      takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
      for more information on Mobilenet.\
      """)
FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

### labelling


from tensorflow_for_poets_2.scripts.label_image import *

model_file = 'tensorflow_for_poets_2\\tf_files\\output_graph_incep_author.pb' 
label_file = 'tensorflow_for_poets_2\\tf_files\\output_labels.txt'
input_height = 299 #299 for inception, 224 for mobilenet
input_width = 299
input_mean = 128
input_std = 128
input_layer = "Mul" #"Mul" for inception, "input" for mobilenet
output_layer = "final_result"

def runlabel(file_name):
 
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

  #print('Evaluation time: {:.3f}s'.format(end-start))
  output_list = []
  #template = "{} (score={:0.5f})"
  for i in top_k:
      temp = [labels[i],results[i]]
      output_list += temp

  return output_list



TEST_FOLDER_PATH = 'C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\test_crop\\'
listfile=os.listdir(TEST_FOLDER_PATH)
outcome = []
for file in listfile:
    if file.endswith('jpg'):
        temp = [file] + runlabel(TEST_FOLDER_PATH+file)
        #print (file)
        #runlabel(TEST_PATH+file)
        #print ()
        outcome.append(temp)

df_o = pd.DataFrame(outcome, columns=['File','p1','p1score','p2','p2score','p3','p3score'])
count=0
for i in range(0,65):
    label=df_o.iloc[i,0].split('_')[0].lower()
    pred=df_o.iloc[i,1]
    if label == pred:
        count=count+1
print (count)
print (len(df_o))
print (count/len(df_o)*100)
    


pd.DataFrame.to_csv(df_o,"outcome"+str(FLAGS.how_many_training_steps)+ '_mobilenet_a_'  +"_crop.csv")

# incep, _substract (threshold=10), 500s, 15%
# incep, _substract (diff), 500s, 15%
# incep, _substract (ath), 500s, 16%
# incep, _threshold, 500s, 27%

# incep, jc, 500s, 73%
# incep, jc + random brightness, 500s, 70%

#mobilenet, author's training set, 500, 80%

import sys
import os
PATH = "C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\"
os.chdir(PATH)

import subprocess
import shlex

arg = shlex.split('python -m tensorflow_for_poets_2.scripts.retrain --image_dir train_crop')
#arg = shlex.split('python -m tensorflow_for_poets_2.test --input xyz')

print(subprocess.Popen(arg,stdout=subprocess.PIPE).stdout.read().decode())

for line in subprocess.Popen(arg,stdout=subprocess.PIPE).stdout.readlines():
    sys.stdout.write(line)
    
