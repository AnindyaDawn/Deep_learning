# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 21:52:47 2020

@author: LENOVO
"""
# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing


# fittng the CNN to the images
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# Part 2 - Building the CNN
#using the VGG16 pretrained model.
from keras import optimizers
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet' , include_top= False , input_shape=(64, 64, 3))
conv_base.summary()
cnn = tf.keras.models.Sequential()
cnn.add(conv_base)
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer = optimizers.RMSprop(lr=2e-5) , loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, batch_size= 32, validation_data = test_set, epochs = 20)


# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, batch_size= 10, validation_data = test_set, epochs = 25)

# Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
