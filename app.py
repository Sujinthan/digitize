"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from flask import request
from flask import jsonify
from PIL import Image, ImageOps
import re
import  base64
from io import BytesIO
import numpy as np
from keras.models import  load_model
from keras.preprocessing.image import img_to_array
from flask import Flask
app = Flask(__name__, template_folder='template')


model = load_model('model.h5')
@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""

    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

def prepare_image(image, target):
    '''
    Preprocess the image and prepare it for classification
    '''
    img_array = Image.open(BytesIO(image)).convert('L')
    img_array = ImageOps.invert(img_array)

    new_array = img_array.resize((target,target))

    img = img_to_array(new_array)
    img = img/255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/imgToText', methods=['POST'])
def imgToText():
    '''
    Image is received as DataURI, it is convereted to Image, and preprocessed.
    The model uses the preprocessed image to make a prediction. 

    returns JSON representation of the model prediction

    '''
    image_b64 = request.values['imageBase64']
    image_data = re.sub('^data:image/.+;base64,', '', image_b64)
    image_data =base64.b64decode(image_data)
    IMG_SIZE =28
    img = prepare_image(image_data, IMG_SIZE)

    #load model
    #model = load_model('model.h5')
    #get class names 
    y_Labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'a', 27: 'b', 28: 'd', 29: 'e', 30: 'f', 31: 'g', 32: 'h', 33: 'n', 34: 'q', 35: 'r', 36: 't'}
    #make prediction
    prediction =model.predict_classes(img)
    return jsonify(y_Labels[prediction[0]])
