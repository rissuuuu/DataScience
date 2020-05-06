from flask import Flask, request,render_template
import util

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])



if __name__ == "__main__":
    app.run()
