from flask import Flask,render_template,requests
import numpy as np
import pandas as pd
from sklearn.externals import joblib
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',method=['GET','POST'])
def predict():
    if requet.method=='POST':
        try:

if __name__ =="__main__":
    app.run()
