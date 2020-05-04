from flask import Flask,render_template,requests
import numpy as np
import pandas as pd
app=Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict',method=['GET','POST'])
def predict():
    if request.method=='POST':
        try:


        except valueError:
    return render_template('predict.html')

if __name__ =="__main__":
    app.run()
