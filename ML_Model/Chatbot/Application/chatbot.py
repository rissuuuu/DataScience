from flask import Flask,render_template,request
import util

app=Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')


@app.route('/predict',methods=['GET','POST'])
def predict_text():
    if request.method=='POST':
        try:
            text=str(request.form.get('uname'))
            predicted=util.response(text)
            return render_template('home.html',prediction=predicted)
        except valueError:
            return "Invalid Values"



if __name__=='__main__':
    app.run(host='0.0.0.0')
