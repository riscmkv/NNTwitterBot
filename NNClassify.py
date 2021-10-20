# tensorflow will be our main library for the neural network
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import to_categorical

import pickle
import glob
import logging
logging.getLogger('tensorflow').disabled = True


def rgb2gray(img):
    return np.dot(img[...,:3],[0.299, 0.587, 0.144])

def crop(image):
    height = image.shape[0]
    width = image.shape[1]
    if (height > width):
        return image[int(height/2 - width/2):int(height/2 + width/2),:]
    else:
        return image[:,int(width/2 - height/2):int(width/2 + height/2)]

def NormalizeImage(fname):
    # open the image
    img = plt.imread(fname)
    # convert to grayscale if necessary
    if(len(img.shape) > 2):
        img = rgb2gray(img)
    # crop the image into a square
    img = crop(img)
    # resize the image
    img = resize(img, (100, 100), anti_aliasing=True)
    #normalize the image
    img -= np.mean(img)
    return img

def PredictImage(fname):
    classifier = load_model('ImageRecognizer.h5')
    labels = pickle.load(open('labels.dat', 'rb'))
    img = np.array([NormalizeImage(fname)[..., np.newaxis]])
    train_label=pickle.load(open('labels.dat', 'rb'))
    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    prediction = classifier.predict(img)
    return le.classes_[np.argmax(prediction)]

def PredictImageWithCertainty(fname):
    classifier = load_model('ImageRecognizer.h5')
    labels = pickle.load(open('labels.dat', 'rb'))
    img = np.array([NormalizeImage(fname)[..., np.newaxis]])
    train_label=pickle.load(open('labels.dat', 'rb'))
    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    prediction = classifier.predict(img)
    return (prediction[0][np.argmax(prediction)], le.classes_[np.argmax(prediction)])

def PredictFolder(fname):
    for file in glob.glob(fname + '/*'):
        print('file:', file, '\t\t result:', PredictImage(file))
