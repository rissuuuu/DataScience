from flask import Flask,render_template,flash,request,url_for,redirect,session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
import keras

IMAGE_FOLDER=os.path.join('static','img_pool')
app=Flask(__name__)
app.config['UPLOAD_FOLDER']=IMAGE_FOLDER

def init():
    global model,graph
    model=load_model('sentiment_analysis.h5')
    graph=tf.get_default_graph()


@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')



if __name__=='__main__':
    app.run()
