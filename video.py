import os
import numpy as np
import cv2

ROOT = "C:\\Users\\peanut\\Dropbox\\KE5108 DEVELOPING INTELLIGENT SYSTEMS\\CA3\\"
os.chdir(ROOT)

from tensorflow_for_poets_2.scripts.label_image import *

# CONFIG FOR LABELLING or PREDICTION
# label_image.py for prediction is small and the default ouput is not 
# suitable for integration with our video demo, video.py. Hence the code is 
# copied and modified here

model_file = 'tensorflow_for_poets_2\\tf_files\\output_graph_mobilenet.pb' 
label_file = 'tensorflow_for_poets_2\\tf_files\\output_labels.txt'
input_height = 224 #299 for inception, 224 for mobilenet
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input" #"Mul" for inception, "input" for mobilenet
output_layer = "final_result"

def run_label(file_name):
 
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

  return labels[top_k[0]]


xml_PATH="C:\\Users\\peanut\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(xml_PATH)

cap = cv2.VideoCapture(0)
time.sleep(2)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    faces = face_cascade.detectMultiScale(frame, 1.05, 5)
    for (x, y, w, h) in faces:
        
        #crop_face,resize,greyscale
        img_crop = frame[y:y+h,x:x+w]
        img_crop = cv2.resize(img_crop, (224,224))
        img_crop = np.dot(img_crop[...,:3], [.114,.587,.299])
        img_crop = img_crop.astype('uint8')
        
        #write to jpg. labelling 
        cv2.imwrite('tmp.jpg',img_crop)        
        text=run_label('tmp.jpg')            
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
        cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
