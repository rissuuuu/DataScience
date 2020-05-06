from flask import Flask, request,render_template
import util
import joblib
file=open('banglore_home_prices_model.pickle','rb')
model=joblib.load(file)
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET','POST'])
def predict_home_price():
    if request.method=='POST':
        try:
            Sqft=float(request.form['Sqft'])
            Bhk=float(request.form['Bhk'])
            Bathroom=float(request.form['Bathroom'])
            location=str(request.form.get('comp'))
            # predicted= float(util.get_estimated_price("cv raman nagar",Sqft,Bhk,Bathroom))
            predicted= util.get_estimated_price(location,Sqft,Bhk,Bathroom)
            return render_template('home.html',prediction=predicted)
        except valueError:
            return "Check the values"

if __name__ == "__main__":
    app.run(host='0.0.0.0')
