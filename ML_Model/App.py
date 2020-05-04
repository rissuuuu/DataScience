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
    if request.method=='POST':
        try:
            NewYork=float(request.form['NewYork'])
            California=float(request.form['California'])
            Florida=float(request.form['Florida'])
            RnDSpend=float(request.form['RnDSpend'])
            AdminSpend=float(request.form['AdminSpend'])
            MarketSpend=float(request.form['MarketSpend'])
            pred_args=[RnDSpend,AdminSpend,MarketSpend,California,Florida,NewYork]

        except valueError:
    return render_template('predict.html')

if __name__ =="__main__":
    app.run()
