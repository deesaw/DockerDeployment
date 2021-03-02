import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import cv2
import os
from pathlib import Path
import swifter

from tensorflow.keras.layers import Conv2D, Flatten, Dense,Input,concatenate,MaxPooling2D
from tensorflow.keras.layers import Activation,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow import keras

import swifter

from scipy import spatial
import warnings
warnings.filterwarnings("ignore")

import pickle

file_to_read = open("resnetvector.pickle", "rb")
d=pickle.load(file_to_read)
file_to_read1 = open("autoencodervector.pickle", "rb")
di=pickle.load(file_to_read1)
conv_autoencoder = keras.models.load_model('conv_autoencoder_deepa_100_autoencoder_accu.h5')
model = Model(conv_autoencoder.input, conv_autoencoder.get_layer('maxpool_3').output)

def get_feature_vector_autoencoder(img):
    img = img * 1./255
    img1 = cv2.resize(img, (60, 60))
    feature_vector = model.predict(img1.reshape(1, 60, 60, 3))
    f1 =np.ravel(np.array(feature_vector, dtype='float64'))
    f1 =f1.reshape(1, -1)
    return f1

def getfeaturevectorforallsourceimagesautoencoder(image):
    image_f=image
    contrast_file = cv2.imread(image_f)
    contrast_file = cv2.resize(contrast_file,(60,60),3)
    f2=get_feature_vector_autoencoder(contrast_file)
    return f2

def calculate_similarity(vector1, vector2):
    return (1-spatial.distance.cosine(vector1, vector2))

vgg16 = keras.applications.ResNet152(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))

basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('avg_pool').output)

##To get feature vector
def get_feature_vector(img):

    img1 = cv2.resize(img, (224, 224))
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
    return feature_vector

def getfeaturevectorforallsourceimages(image):
    image_f=image
    contrast_file = cv2.imread(image_f)
    contrast_file = cv2.resize(contrast_file,(224,224),3)
    f2=get_feature_vector(contrast_file)
    return f2

def displayimage(imgpath):
    pil_im = Image.open(imgpath, 'r')
    plt.figure()
    plt.imshow(np.asarray(pil_im))

def resnet_search(imagename,master):
    Sourceimage=imagename
    f1=getfeaturevectorforallsourceimages(Sourceimage)
    #displayimage(Sourceimage)
    #slave = patho
    l=[]
    dilr={}
    for ed in d:
        similarity=calculate_similarity(f1,d[ed])
        if similarity > 0.7:
            #print(ed,':',similarity)
            pathofimage=os.path.join(master,ed)
            l.append(ed)
            dilr[ed]=similarity
    return l,dilr
def autoencoder_search(imagename,master):
    Sourceimage=imagename
    f1=getfeaturevectorforallsourceimagesautoencoder(Sourceimage)
    l=[]
    dil={}
    for ed in di:
        similarity=calculate_similarity(f1,di[ed])
        if similarity > 0.93:
            #print(ed,':',similarity)
            pathofimage=os.path.join(master,ed)
            l.append(ed)
            dil[ed]=similarity
    return l,dil
def slideSimilarity():
    path = Path(os.getcwd())
    path=path.parent
    master = os.path.join(path,"static")
    master = os.path.join(master,"Source")
    uploaded_file='uploads_f\\'
    slave = os.path.join(path,uploaded_file)
    for file_cont in os.listdir(slave):
        if file_cont.split('.')[1]=='jpg':
            filename=os.path.join(slave,file_cont)

    result_autoencodera,dil=autoencoder_search(filename,master)
    result_autoencoderaa = sorted(dil.items(), key=lambda x: x[1], reverse=True)
    print(result_autoencoderaa)
    result_autoencoder=[]
    for i in result_autoencoderaa:
	      result_autoencoder.append(i[0])
    result_resneta,dilr=resnet_search(filename,master)
    result_resnetaa = sorted(dilr.items(), key=lambda x: x[1], reverse=True)
    print(result_resnetaa)
    result_resnet=[]
    for i in result_resnetaa:
        result_resnet.append(i[0])
    return result_resnet,result_autoencoder
