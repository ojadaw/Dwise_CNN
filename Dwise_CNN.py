'''===================================================================================
This code is a project to design a CNN that classifies multiple images.
I believe this is not a new invention or something new out of the box. It is 
also not a reproducible code just taken from someone, but actually using the subjective 
idea of the topics tought in the Pattern Recognition class ECE2372 at the University of Pittsburgh, PA, USA. The author of this code are students in the class, and this code is
intended to portray their learning experiences in the ideas provided by their great instructor Dr. Dallal. There is no specific license to use this code. However, do not forget to provide credit, or citation to code when used in your research or work.

AUTHORS: Collins Dawson, Sr. 
	 Ph.D Student 
	 University of Pittsburgh, PA
	 SHREC LAB - Research Assistant

	Travis Wise
	MS Student	 
	University of Pittsburgh, PA
	SHREC LAB - Research Assistant
==================================================================================='''

'''----------- HERE WE GO! ----------------
We start with importing all the needed libraries into python
--------------------------------------------------'''
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras import models
from keras import layers	
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras import backend as Ker
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
from PIL import Image
import glob
from sklearn.model_selection import cross_val_score

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


''' ------------------ IT'S DATASET DOWNLOAD TIME! ------------------------
We downloaded the datasets from two sources...
1. We downloaded the cats and dogs images from kaggle
2. And then downloaded images from the imagenet/cifar-10 database.
The training data is save in the train folder, while the test dataset is
saved in the test folder.

Creating a function that reads the downloaded images and 
preprocesses the images for the classifier model
-----------------------------------------------------------------------------'''
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import os
from os.path import join
#----------------
train1_file = 'train1' # Assign the training images to a variable
test_file = 'test'  # Assign the testing images to a variable
#path, dirs, files =os.walk('train1').next()
train1_file = [join(train1_file, filenum) for filenum in ['1.jpg',
							 '2.jpg',
							 '3.jpg',
							 '4.jpg',
							 '5.jpg',
							 '6.jpg',
							 '7.jpg']]


image_size = 224

def read_and_prep_images(train1_file, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in train1_file]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)
#-------------------------------------------------
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from IPython.display import Image, display
import matplotlib.image as mping
#sys.path.append('/kaggle/input/python-utility-code-for-deep-learning-exercises/utils')
#from decode_predictions import decode_predictions
#from tensorflow.python.keras.application import ResNet50

 # ----Using the ResNet50 CNN for and imagenet dataset for training
Dwise_model = ResNet50(weights = 'imagenet')
#----------------------------------------------------
test_Dwise_model = read_and_prep_images(train1_file)
predict_Dwise = Dwise_model.predict(test_Dwise_model)
Dwise_decode = decode_predictions(predict_Dwise, top=3)

for i, img_path in enumerate(train1_file):
	img = mping.imread(img_path)
	plt.imshow(img)
	plt.show()
	print (Dwise_decode[i])

'''--------------------
Coding the CNN model without using already made network, such as ResNet as done above
-----------------------------------------------------'''
#(x_train, y_train),(x_test, y_test) = train1_file.load_data()
def Dawise_CNN():
	Daw_CNN = Sequential()
	Daw_CNN.add(Conv2D(30, (5, 5), input_shape = (1, 28, 28), activation = 'relu'))
	Daw_CNN.add(MaxPooling2D(pool_size=(2, 2)))
	Daw_CNN.add(Conv2D(15, (3, 3), activation='relu'))
	Daw_CNN.add(MaxPooling2D(pool_size=(2, 2)))
	Daw_CNN.add(Dropout(0.2))
	Daw_CNN.add(Flatten())
	Daw_CNN.add(Dense(128, activation='relu'))
	Daw_CNN.add(Dense(50, activation='relu'))
	Daw_CNN.add(Dense(num_classes, activation='sigmoid'))
	# Compile model
	Daw_CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return Daw_CNN
#--------------------------------------------------------------------
Daw_CNN = KerasClassifier(build_fn = Dawise_CNN, batch_size =10, nb_epoch =100)
accuracies = cross_val_score(estimator = Daw_CNN, X = train1_file, cv =7, n_jobs = -1)

Dawise_scores = Daw_CNN.evaluate(train1_file, verbose = 0)
print('Daw_CNN Error rate is:  %0.2f%%' % (100 - Dawise_score[1]*100))




