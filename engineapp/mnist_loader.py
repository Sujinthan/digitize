import numpy as np
import pandas as pd
import gzip
import cPickle
import scipy.io as sio
from PIL import Image
from skimage.filter import threshold_otsu
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import random
from random import shuffle

def loadData():

    df = pd.read_csv('labels.csv',header=0)
    X = []
    X+=( [get_img(i,df, 28) for i in df.index])
    Y = np.asarray(df['ASCII'])
    indices = list(zip(X, Y))
    shuffle(indices)

    new_indices = np.array(indices)
    train = indices[:65800]
    test = indices[65800:]
    train_inputs = [np.reshape(x[0], (784, 1))for x in train ]
    training_results = [vectorized(y[1]) for y in train]
    training_data = zip(train_inputs, training_results)
    test_inputs = [np.reshape(x[0], (784,1)) for x in test]
    te_d = []
    for item in test:
        #print(chr(item[1]))
        te_d.append(item[1])
    te_d = np.array(te_d)
    test_data=zip(test_inputs, te_d)
    return (training_data, test_data)

def vectorized(j):
    new_dict = np.zeros((128, 1))
    new_dict[j] = 1.0
    return new_dict

def get_img(i,df, size):
    """
    Returns a binary image from my file directory with index i
    """
    tem_size = 28, 28
    #print(df.at([0], ['Location']))
    img = Image.open(df.ix[i, 'Location'] + '.png')
    img = img.convert("1")
    img = img.resize((size,size))
    #img.thumbnail(tem_size,Image.ANTIALIAS)
    x =list(img.getdata())
    new_x = np.array(x,dtype=np.float)
    return new_x

if __name__ == "__main__":
    loadData()
    print("this is load_data()")
    load_data_wrapper()
