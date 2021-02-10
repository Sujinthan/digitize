"""
A Convolutional Neural Network class that recognizes handwritten letters. 
"""

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Activation, MaxPool2D, AvgPool2D, Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm
from keras.regularizers import l2, l1
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import itertools
import os
import shutil
import cv2
import random
import pickle
import math
import gc
import tensorflow as tf
import tensorflow_datasets as tfds
from extra_keras_datasets import emnist
from PIL import Image

from scipy.io import loadmat
from keras.utils import np_utils
import idx2numpy


#Enable XLA
tf.config.optimizer.set_jit(True)
#Return a list of GPU devices
gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    #set Tensorflow to not allocate all memory on the selected GPU device
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


class CNN():
    
    def __init__(self):
        self.train_path = "'Images for CNN/imageTest/Output/balanced/train'"
        self.test_path = "Images for CNN/imageTest/Output/balanced/test"

    def train(self):
        batch = 64
        #create more images using data agumentation 
        train_datagen = ImageDataGenerator(
            rotation_range=10, zoom_range=0.10, width_shift_range=0.10, rescale=1./255, height_shift_range=0.10)
        test_datagen = ImageDataGenerator(rescale=1./255)

        #build CNN
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1),
                         activation="relu", strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))


        model.add(Conv2D(64, (5, 5), activation="relu",
                         strides=2, padding='same'))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (4, 4), activation="relu"))
        model.add(BatchNormalization())

        model.add(Flatten())

        model.add(Dropout(0.4))

        model.add(Dense(37, activation='softmax'))

        model.summary()
        #set learning rate 
        lrate = 0.01
        opt = keras.optimizers.SGD(lr=lrate)
        epochs = 100
        #reduce learning rate when val_loss stops improving
        annealer = ReduceLROnPlateau(
            monitor='val_loss', mode='min',  factor=0.95, patience=3, verbose=1)
        #compile with SGD optimizer and cross entropy cost 
        model.compile(loss="categorical_crossentropy",
                      optimizer=opt, metrics=['accuracy'])
        #log each epoch to file
        csv_logger = CSVLogger('training.log')
        #stop training when val_loss stops improving
        es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
        #save model on lowest val_loss
        mc = ModelCheckpoint('model.h5', monitor='val_loss',
                             mode='min', save_best_only=True)
        #train the model
        model.fit(train_datagen.flow_from_directory(self.train_path, target_size=(28, 28), batch_size=batch, color_mode='grayscale', class_mode='categorical', shuffle=True), epochs=epochs,
                  validation_data=test_datagen.flow_from_directory(self.test_path, target_size=(28, 28), batch_size=batch, color_mode='grayscale', class_mode='categorical', shuffle=False), callbacks=[es, mc, annealer, csv_logger])
        for i in range(1, 5):
            #reduce the learning rate by 10
            if (i % 2 == 0):
                lrate = lrate / 10
            #clear clutter of old models and layers to for memory
            K.clear_session()
            del model
            
            tf.config.optimizer.set_jit(True)
            gpus = tf.config.experimental.list_physical_devices('GPU')
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass
            
            #load model to train
            model = load_model('model.h5')
            model.compile(loss="categorical_crossentropy",
                          optimizer=opt, metrics=['accuracy'])
            model.fit(train_datagen.flow_from_directory(self.train_path, target_size=(28, 28), batch_size=batch, color_mode='grayscale', class_mode='categorical', shuffle=True), epochs=epochs, validation_data=test_datagen.flow_from_directory(self.test_path, target_size=(28, 28), batch_size=batch, color_mode='grayscale', class_mode='categorical', shuffle=False),
                      callbacks=[es, mc, annealer,  csv_logger])
        #save final model
        model.save("model_test.h5")

    def test(self):
        '''
        Test the model using images from differect directory, print results.
        '''
        #load model
        model = load_model('model.h5', compile=True)
        test_datagen = ImageDataGenerator(rescale=1./255)
        #load images that is different from validation and train images
        train_generator = test_datagen.flow_from_directory('Images for CNN/imageTest/Output/balanced/test', target_size=(28, 28), batch_size=32,
                                                           color_mode='grayscale', class_mode='categorical', shuffle=False)
        test_loss, test_acc = model.evaluate_generator(
            train_generator, verbose=1)
        #print test_loss and test_acc
        print('\nTest Loss: ', test_loss)
        print('\nTest accuracy: ', test_acc)

    def predict(self, path):
        '''
        Predict image from given path.  
        '''
        #set image size
        IMG_SIZE = 28 
        #Open image and convert to grayscale
        img_array = Image.open(path).convert("L")
        #resize image to be 28x28
        new_array = np.reshape(img_array, (IMG_SIZE, IMG_SIZE, 1))
        im2arr = np.array(new_array)
        im2arr = im2arr.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        #load model
        model = load_model('model.h5')
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = test_datagen.flow_from_directory('Images for CNN/imageTest/Output/balanced/training', target_size=(28, 28), batch_size=32,
                                                           color_mode='grayscale', class_mode='categorical', shuffle=False)
        #get class names 
        y_Labels = train_generator.class_indices
        #switch key, value for faster way to get image category
        y_Labels = {y: x for x, y in y_Labels.items()}
        print(type(y_Labels))
        #make prediction
        prediction = model.predict_classes([im2arr])
        print(type(prediction))
        return y_Labels[prediction[0]]


#if __name__ == "__main__":

#    temp = CNN()
#    # temp.train()
#    # temp.guess()

#    print("This is prediction")
#    print(temp.predict())
#    print("------")
#    print("Done!!!")
