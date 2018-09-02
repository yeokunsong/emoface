# Motivation
This project is inspired by a student project from Stanford University's EE368 module. Facial emotion images from CK+ dataset were used to train a saptial feature-based classifier to recognition 7 different types of facial emotion.
J Pao. Emotion Detection Through Facial Feature Recognition (#37). . https://web.stanford.edu/class/ee368/Project_Autumn_1617/Reports/report_pao.pdf. 


# Pre-requisites:
1. Install Python 3.6.* (Tensorflow not support 3.7 yet)
2. Install OpenCV
3. Install Numpy
4. Install Tensorflow-GPU for 3.6.*
5. Install other packages used in the python files

# Image source
1. Unzip the original images from CK+ dataset  (http://www.consortium.ri.cmu.edu/data/ck/; registration required) to "extended-cohn-kanade-images\cohn-kanade-images"
2. Unzip the original labels from CK+ dataset to "Emotion_labels\Emotion"

# Extract and preprocess images
This step can be skipped if you just use the processed datasets in the zip files

1. Run "data_extraction_preprocessing.py". Training and testing folders will be created. "crop_face.py", "subtract_neutral_face.py" and "threshold_face.py" are scripts containing necessary modules.

# Check differences in training and testing images between Pao's dataset and our datasets
1. Open and run "find_difference.py"

# Model training and labelling
1. tensorflow_for_poets_2\ should be in root
2. Open and edit the CONFIG lines in "run_cnn.py"
3. Run the file
4. If another model is to be trained, the kernel should be restarted first to clear the cache.
5. Raw results are commented in the script after the codes

# Webcam implementation
1. Open and run "video.py"
