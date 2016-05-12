# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 20:34:01 2016

@author: Suji
"""

from flask import Flask, request,jsonify
#import Network as Net
import random
import numpy as np
from PIL import Image as pic
import base64
import traceback
#### Libraries
# Standard library
import pickle
import gzip


class HMM(object):
    def __init__(self):
        sefl.N = 2
        self.M = 27
        self.pi = np.array([0.5, 0.5])
        self.A = np.array([[0.5, 0.5],[0.5,0.5]])
        self.B = np.array([[1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,
        1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,
        1/27,1/27,1/27],[1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,
        1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,1/27,
        1/27,1/27,1/27]])
        T = []

    def forward(self, a):
        #hope this works
        c = 0
        len_of_observ = len(a)
        for i in range(0, self.N) :
            self.T[i] = self.pi[i]*self.B[i]*a[i]
            c += self.T[i]

        #Scale the T
        new_c = 1/c
        for i in range (0, self.N):
            self.T[i] = new_c * T[i]

        for t in range (1, len_of_observ):
            ct = 0
            for i in range (0, self.N):
                total = 0
                for j in range (0, self.N):
                    total += self.T[i] *self.A[i][j]
                self.T[i] = T[i]*self.B[i]*a[i]
                ct += total
            ct = 1/ct

            for i in rang(0, self.N):
                self.T[i]= ct*self.T[i]

        return self.T;

#Leaving this function empty to see if class works without it
    def viterbi (self, a):
        v = [{}]
        for i in range (0, self.N):
            v[0][i]= self.pi[i]*self.B[i]
        for t in range (2, self.T):
            denom = 0
            for s in range(0, to self.N):
                denom = V[s] + self.A[t][s]*


    def baum_welch(self, training_data):


    '''def __init__(self, sizes):
        self.num_layer = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes [1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        np.seterr(all="ignore")
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):

        list_training_data = list(training_data)
        if test_data:
            list_test_Data = list(test_data)
            n_test = len(list_test_Data)
        n = len(list_training_data)
        for j in range(epochs):
            random.shuffle(list_training_data)
            mini_batches = [list_training_data[k:k+mini_batch_size]for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                evulate = self.evaluate(list_test_Data);
                print("Epoch{0}:{1}/{2}" .format( j, evulate, n_test))
            else:
                print("Epoch {0} complete". format(j))

    def backdrop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-1-1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backdrop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w ,nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)for (x, y) in test_data]
        final = sum(int(x==y)for (x,y) in test_results)
        return final

    def cost_derivative(self, output_activatoins, y):
        return (output_activatoins - y)

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

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
    training_data, validation_data, test_data = pickle.load(f)
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
    training_data = zip(training_inputs, training_results)

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

def result (data):
    result = [vectorized_result(y) for y in data[0]]
    return result

Net = Network([784, 100,10])'''

def image_function(image):
    size = 28,28
    image_bw = pic.open(image)
    image_bw.convert("1")
    image_bw.thumbnail(size,pic.ANTIALIAS)
    x =list(image_bw.getdata())
    new_x = np.resize(x,(784,1))
    t = np.argmax(Net.feedforward(new_x))
    return (t)

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def hello_world():
    received_img = request.data
    #decode_Imge = pic.open(BytesIO(base64.b64decode(received_img)))
    imgdata = base64.b64decode(received_img)
    filename = 'some_image.bmp'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)
    resutl = image_function(filename)

    return jsonify(results = resutl)

if __name__ == '__main__':
    Net = Network([784, 100,10])
    training_data, validation_data,test_data = load_data_wrapper()
    Net.SGD(training_data, 30, 10, 5.0, test_data = test_data)
    app.run(host ="192.168.0.20", threaded=True)
