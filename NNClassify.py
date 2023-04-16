# tensorflow will be our main library for the neural network
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
import glob
import logging
import random
from twython import Twython
import pytumblr
import json
import confidence
import resnext
import os
from urllib import request
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras.utils import to_categorical


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

#creates log file and sets log variables
logger = logging.getLogger('Image Bot')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('ImageBot.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(logger)

#some auth bullshit copied from formerly twitterBotAPI.py

#twitter API stuff
twitter = Twython(
	'consumer_key',
	'consumer_secret',
	'access_token',
	'access_token_secret'
)

#tumblr API stuff
tumblr = pytumblr.TumblrRestClient(
    '<consumer_key>',
    '<consumer_secret>',
    '<oath_token>',
    '<oath_secret>'
)

#defines legal image extensions and the URL for r/cursedimages
img_ext = ['jpg', 'png', 'jpeg']


def isImageURL(post):
    for ext in img_ext:
        if(post['data']['url'].endswith(ext)):
            return True
    return False




#gets images from reddit
#filters out 18+ images
#randomly chooses one image
def GetRandomImage():
    json_dat = pickle.load(open('reddit_info.json', 'rb'))

    imgurl = []
    posts = json_dat['data']['children']
    for post in posts:
        imgurl.append(post['data']['url'])

    rand_url = random.sample(imgurl, 1)[0]
    request.urlretrieve(rand_url, './random_img/temp_img.' + rand_url.split('.')[-1])
    return './random_img/temp_img.' + rand_url.split('.')[-1]

#defines the submitter name class for twitter
def getSubmitterName(fname):
    if(len(fname.split('.')) < 3):
        return 'none'
    else:
        return fname.split('.')[-2]
    
#defining when the queue is empty and what to do
def QueueIsEmpty():
    submissions = glob.glob("./submissions/*")
    curated = glob.glob("./curated/*")
    
    if (len(submissions) != 0) or (len(curated) != 0):
        return False
    else:
        return True

#defining images in class GetImage
#separating the cream of the crop from the stupid people (submitters and reddit)
def GetImage(fanSubmit=False):
    submissions = glob.glob("./submissions/*")
    if fanSubmit:
        submissions = glob.glob("./fan_submit/*")
    curated = glob.glob("./curated/*")
    
    if(len(submissions) != 0):
        img_path = random.sample(submissions, 1)[0]
        submitter = getSubmitterName(img_path)
        return [img_path, submitter]
    elif(len(curated) != 0):
        img_path = random.sample(curated, 1)[0]
        return [img_path, 'none']
    else:
        return [GetRandomImage(), 'none']
    
#defining the class to remove an image from the queue after its been posted
def RemoveFromQueue(img_path):
    fname = img_path.split('\\')[-1]
    os.remove(img_path)

#defining a class to list all saved images in the folder
def list_saved_images(folders):
    files = []
    for folder in folders:
        files = files + glob.glob("./" + folder + "/*")
    for i in range(len(files)):
        files[i] = files[i].split('\\')[-1]
    return files

#defining a class to mention the submitter on twitter (not to be used elsewhere due to the complexity of the API and the fact that only twitter is requested)
def get_screen_name(userid, twitter):
    uname_response = twitter.request(
        'https://api.twitter.com/1.1/users/show.json',
        method='GET',
        params={'user_id' : userid},
        version='1.1'
    )
    return uname_response['screen_name']

#this actually looks like a meme but okay riscy you do you with your pickles and haystacks
def is_in_list(needle, haystack):
    for straw in haystack:
        if needle == straw:
            return True
    return False

folders = ['submissions', 'submissions_completed', 'submissions_rejected', 'submissions_unmoderated']

#a lot to unpack here
#first off, redefines confidence into a list
#builds a message text out of the confidence calculations
def gen_tweet_string(prediction, img_path):
    message = None
    confidence_calced = [None, None, None, None]
    confidence_calced[0] = confidence.calc_confidence_idx(img_path, prediction[2][0])
    confidence_calced[1] = confidence.calc_confidence_idx(img_path, prediction[2][1])
    confidence_calced[2] = confidence.calc_confidence_idx(img_path, prediction[2][2])
    confidence_calced[3] = confidence.calc_confidence_idx(img_path, prediction[2][3])

    message = "Image prediction:\n"
    message += prediction[0][0] + " (" + str(round(confidence_calced[0]*100, 2)) + "%)\n"
    message += prediction[0][1] + " (" + str(round(confidence_calced[1]*100, 2)) + "%)\n"
    message += prediction[0][2] + " (" + str(round(confidence_calced[2]*100, 2)) + "%)\n"
    message += prediction[0][3] + " (" + str(round(confidence_calced[3]*100, 2)) + "%)"