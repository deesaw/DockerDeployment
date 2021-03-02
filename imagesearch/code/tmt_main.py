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

#from keras.layers import Flatten, Dense, Input,concatenate
#from keras.layers import Conv2D, MaxPooling2D
#from keras.layers import Activation, Dropout
#from keras.models import Model
#from keras.models import Sequential
#import tensorflow as tf
from scipy import spatial
import warnings
warnings.filterwarnings("ignore")

import pickle
from flask import *
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import swifter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle as pk
from sklearn.metrics import confusion_matrix, accuracy_score,average_precision_score,classification_report,f1_score
#import urllib.parse
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import time
#from flask_restful import Api,Resource
from flask import jsonify
import json
import os,shutil
import zipfile
import glob
import tensorflow as tf
import ImageSearch as slideSim
import time
import swifter
from threading import Thread as t
import queue
from pathlib import Path

UPLOAD_FOLDER = '../uploads_f/'#'/uploads_f'
ALLOWED_EXTENSIONS = set(['jpg'])
pd.set_option('display.max_colwidth', -1)
app = Flask(__name__,template_folder='../templates',static_folder='../static')
#app=Flask(__name__,template_folder='../templates',static_folder='../static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2LF4Q8z'
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
global recent_upload



@app.route('/home', methods=['GET', 'POST'])

def upload_file():
    if request.method == 'POST':
        print(os.getcwd())
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        print('******************************')
        print(allowed_file(file.filename))
        if allowed_file(file.filename)== False:
            flash('Please upload jpg image')
        if file and allowed_file(file.filename):
            filelist = [ os.remove(os.path.join(app.config['UPLOAD_FOLDER'],f)) for f in os.listdir('../uploads_f/') ]
            print(filelist)
            #for f in filelist:
            #    os.remove(f)
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


    f_list = [f for f in listdir('../uploads_f/') if isfile(join('../uploads_f/', f))]

    return render_template('upload.html',file_list = f_list)

@app.route('/blog/<int:postID>')
def show_blog(postID):
   return 'Blog Number %d' % postID

@app.route("/preview")
def preview(filename):
    return 'File Uploaded'

@app.route("/predict")
def predict():
    return render_template('predict.html')

@app.route('/function')
def get_ses():
    print("server has called api")
    r,a = slideSim.slideSimilarity()
    #h = m['Source Image']
    return render_template('upload0.html',r=r,a=a)

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0',debug=True,port=5000)
