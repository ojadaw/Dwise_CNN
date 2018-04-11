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
# Convolutional Neural Network
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
Dawson_classifier = Sequential()
# Step 1 - Convolution
Dawson_classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
Dawson_classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
Dawson_classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
Dawson_classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
Dawson_classifier.add(Flatten())
# Step 4 - Full connection
Dawson_classifier.add(Dense(units = 128, activation = 'relu'))
Dawson_classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN
Dawson_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
import datetime as dt

Wise_train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

Wise_test_datagen = ImageDataGenerator(rescale = 1./255)

Wise_training_set = Wise_train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

Wise_test_set = Wise_test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
start_time = dt.datetime.now() # Setting up a time to capture the duration for training

Dawson_classifier.fit_generator(Wise_training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = Wise_test_set,
                         validation_steps = 2000)

print('It takes %s to train this network' % (dt.datetime.now() - start_time()))                         
print(' ')
#print('The error is: %s ' % binary_crossentropy)
