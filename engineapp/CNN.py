"""
A Convolutional Neural Network class that recognizes handwritten letters. 
"""

import numpy as np

import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Activation, MaxPool2D, Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import CSVLogger, LearningRateScheduler, EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import itertools
import os, shutil
import cv2
import random
import pickle

class CNN():
    def __init__(self):
        self.train_path = "/mnt/e/Images for CNN/train"
        self.test_path = "/mnt/e/Images for CNN/test"

    def set_training_data(self):
        img_size = 50
        test_data=[]
        test_X = []
        test_Y = []
        for category in self.lables:
            path = os.path.join(self.test_path, category)

            for img in os.listdir(path):

                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array,(img_size, img_size) )
                    test_data.append([new_array, class_num])
                except Exception as e:
                    pass
        random.shuffle(test_data)
        for features, label in  test_data:
            test_X.append(features)
            test_Y.append(label)
        test_X = np.array(test_X).reshape(-1, img_size, img_size, 1)
        np.save("testX.npy", test_X)
        np.save("testY.npy", test_Y)
    def train(self):
        batch = 64
        train_datagen = ImageDataGenerator(rotation_range=10,  zoom_range = 0.10,  width_shift_range=0.1, height_shift_range=0.1, rescale=1./255, validation_split=0.2)
        #X = np.load(open("trainX.npy"))
        #test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory('Images for CNN/train', target_size=(28,28), batch_size=batch, color_mode='grayscale', class_mode='categorical', subset='training')
        test_generator = train_datagen.flow_from_directory('Images for CNN/train', target_size=(28,28), batch_size=batch, color_mode='grayscale', class_mode='categorical', subset='validation')

        model = Sequential()

        model.add(Conv2D(8, (3,3), input_shape = (28,28, 1), activation="relu" ))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #model.add(MaxPool2D(pool_size=(2,2), strides=2, padding="valid"))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.4))

        model.add(Conv2D(8, (3,3),  activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2), padding="valid"))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.4))


        model.add(Conv2D(8, (5,5),activation="relu", padding='same', strides=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #model.add(MaxPool2D(pool_size=(2,2),strides=2, padding='valid'))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.4))

        #model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid"))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.4))

        model.add(Conv2D(16, (3,3), activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2), padding="valid"))
        #model.add(Dropout(0.4))

        model.add(Conv2D(16, (3,3), activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #model.add(Conv2D(64, (5,5), activation="relu"))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.4))

        #model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid"))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.4))

        #model.add(Conv2D(64, (5,5) ,activation="relu"))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.4))
        model.add(Conv2D(16, (5,5), activation="relu", padding='same', strides=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))
       	#model.add(BatchNormalization())
        #model.add(Dropout(0.4))

        #model.add(Conv2D(32, (4,4), activation="relu"))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.4))
        model.add(Conv2D(32, (3,3), activation="relu"))
       	model.add(BatchNormalization())
        model.add(Dropout(0.4))
        #model.add(Conv2D(128, (1,1), activation="relu"))
        #model.add(MaxPool2D(pool_size=(1,1)))
        #odel.add(BatchNormalization())
        #model.add(Dropout(0.4))
        #model.add(MaxPool2D(pool_size=(1,1), strides=(2,2)))

        model.add(Flatten())
        #model.add(Dropout(0.4))
        #model.add(BatchNormalization())

        model.add(Dense(32, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #model.add(Dense(32,activation="relu"))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.4))

        model.add(Dense(52, activation='softmax'))

        model.summary()
        adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
        epochs = 100
        annealer = LearningRateScheduler(lambda x: 1e-2 * (0.99 ** x))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])
        #datagen = ImageDataGenerator(rotation_range=10,  zoom_range = 0.10,  width_shift_range=0.1, height_shift_range=0.1)
        csv_logger = CSVLogger('training.log')
        es = EarlyStopping(monitor='val_loss', mode='min', patience=25)
        mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', save_best_only=True)
        model.fit_generator(train_generator, epochs =epochs, validation_data=test_generator, callbacks=[es, mc,csv_logger], steps_per_epoch =638955//batch, validation_steps = 159708//batch)
        model.save("model_test.h5")

    def continueTrain(self):
        model = load_model('model.h5')
        testX = np.load(open("testX.npy"))
        testY = np.load(open("testY.npy"))
        csv_logger = CSVLogger('training.log')
        model.fit(testX, testY, batch_size =64, epochs = 25, validation_split=0.1, callbacks=[csv_logger])
        model.save("model.h5")

    def guess(self):
        model = load_model('model.h5')
        img_array = cv2.imread('test9.png', cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (28,28))
        new_array = new_array.reshape( 1, 28, 28, 1)
        outsource = model.predict(new_array)
        return outsource
if __name__ == "__main__":
    #print(trainX[0][0])
    #trainY = np.load(open("trainY.npy"))
    #print(trainY)
    temp = CNN()
    temp.train()
    #temp.continueTrain()
    prediction = temp.guess()
    print("This is prediction")
    print(labels[np.argmax(prediction)])

    #indxPredict = int(np.where(prediction==maxPredict)[1])
    #print(lables[indxPredict])
    #score = temp.test()
    #print(score)
    #print("Score is:")
    print("Done!!!")
    # X = np.load("training.npy")
    # print(X[1][0]j
