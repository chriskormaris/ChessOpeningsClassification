import os

import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from keras.src.saving import load_model
from wtforms import FileField, SubmitField

chess_openings_dict = {
    0: 'Ruy Lopez',
    1: 'Italian Game',
    2: 'Queen\'s Gambit',
    3: 'Sicilian Defense',
    4: 'Nimzo-Indian Defense'
}


def return_prediction(model, chess_opening_image):
    chess_opening_image_array = np.array(chess_opening_image)
    chess_opening_image_array = np.expand_dims(chess_opening_image_array, axis=2)
    chess_opening_image_array = np.expand_dims(chess_opening_image_array, axis=0)
    class_index = np.argmax(model.predict(chess_opening_image_array), axis=1)[0]

    return chess_openings_dict[class_index]


app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = '2afe16a6d630d94cd07c68d5e35568655bf5f60bef29c4f1321fc857816afec9'
app.config['UPLOAD_FOLDER'] = 'static/images'

# REMEMBER TO LOAD THE MODEL!
chess_openings_model = load_model('chess_openings_model.h5')


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class ChessOpeningForm(FlaskForm):
    chess_opening = FileField('Chess Opening')
    submit = SubmitField('Predict')


@app.route('/', methods=['GET'])
def index():
    form = ChessOpeningForm()
    return render_template('index.html', form=form)


@app.route('/prediction', methods=['POST'])
def prediction():
    file = request.files['chess_opening']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
    file.save(file_path)

    chess_opening_image = Image.open(file)
    chess_opening_image = chess_opening_image.resize((100, 100))  # reduce dimensions to 100x100
    chess_opening_image = chess_opening_image.convert('L')  # convert to grayscale
    prediction = return_prediction(model=chess_openings_model, chess_opening_image=chess_opening_image)

    return render_template('prediction.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
