from flask import Flask, request, Response,render_template
from werkzeug.utils import secure_filename

from db import db_init, db
from models import Img
import os

UPLOAD_FOLDER='./images'
app = Flask(__name__)
# SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = r'C:\Users\rissu\A_DataScience\Flask file upload\images'
db_init(app)


@app.route('/')
def hello_world():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload():
    pic = request.files['pic']
    if not pic:
        return 'No pic uploaded!', 400

    filename = secure_filename(pic.filename)
    print(filename)
    mimetype = pic.mimetype
    print(mimetype)
    if not filename or not mimetype:
        return 'Bad upload!', 400
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pic.save(path)

    img = Img(img=pic.read(), name=filename, mimetype=mimetype)
    db.session.add(img)
    db.session.commit()

    return path, 'Img Uploaded!', 200


@app.route('/<int:id>')
def get_img(id):
    img = Img.query.filter_by(id=id).first()
    if not img:
        return 'Img Not Found!', 404

    return Response(img.img, mimetype=img.mimetype)

if __name__ == '__main__':
   app.run()