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
            NewYork=float(request.form['NewYork'])
            Florida=float(request.form['Florida'])
            California=float(request.form['California'])
            RnDSpend=float(request.form['RnDSpend'])
            AdminSpend=float(request.form['AdminSpend'])
            MrktngSpend=float(request.form['MrktngSpend'])
            pred_args=[RnDSpend,AdminSpend,MrktngSpend,California,Florida,NewYork]

            pred_args_arr=np.array(pred_args)
            pred_args_arr=pred_args_arr.reshape(1,-1)

            ml_f=open('linear_reg.pkl','rb')
            ml_model=joblib.load(ml_f)
            model_predict=ml_model.predict(pred_args_arr)
            model_predict=round(float(model_predict),2)
        except valueError:

    return render_template('predict.html')

if __name__ =="__main__":
    app.run()
