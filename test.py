import os
import time
import argparse

PATH = "C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\"
os.chdir(PATH)

with open('test.txt','a+') as f:
    f.write(str(time.localtime().tm_min) + '_' + str(time.localtime().tm_sec) + '\n')
    
print ('This is a test')


parser = argparse.ArgumentParser()
parser.add_argument(
        '--input',
        type=str,
        default='Default string', #train\\
        help='Path to folders of labeled images.'
        )
FLAGS = parser.parse_args()
print (FLAGS.input)


time.sleep(2)
print (FLAGS.input)

#Test with cmd command 
#    py test.py
#    py test.py --input "something"