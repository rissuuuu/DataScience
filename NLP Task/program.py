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
            text=str(request.form.get('question'))
            predicted=util.response(text)
            return render_template('home.html',prediction=predicted)
        except:
            return "Invalid Values"

app.run()