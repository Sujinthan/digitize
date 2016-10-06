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

def load():

    df = pd.read_csv('trainLabels.csv', header=0)
    X =[]
    X+=([get_img(i, 28) for i in df.index])
    Y = np.asarray(df['Class'])
    # method for index-shuffle & dataset split
    indices = list(zip(X, Y))
    new_indices = np.array(indices)
    train = new_indices[:3142]
    test = new_indices[3142:]
    train_inputs = [np.reshape(x[0],(784, 1)) for x in train]
    training_results = [vectorized(ord(y[1])) for y in train]
    training_data = zip(train_inputs, training_results)#<type 'list'>
    test_inputs = [np.reshape(x[0], (784, 1)) for x in test]
    te_d = []
    for item in test:
        te_d.append(ord(item[1]))
    test_data = zip(test_inputs, te_d)
    return (training_data,test_data)

def vectorized(j):
    new_dict = np.zeros((128, 1))
    new_dict[j] = 1.0
    return new_dict
def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)#<type 'list'>
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def get_img(i, size):
    """
    Returns a binary image from my file directory with index i
    """
    tem_size = 28, 28
    img = Image.open('train/'+ str(i+1) + '.Bmp')
    img = img.convert("1")
    img = img.resize((size,size))
    #img.thumbnail(tem_size,Image.ANTIALIAS)
    x =list(img.getdata())
    new_x = np.array(x,dtype=np.float)
    return new_x
