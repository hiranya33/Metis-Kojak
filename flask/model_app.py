from flask import Flask, abort, render_template, jsonify, request
from model_api import get_input,make_prediction
from flask import send_from_directory
import time
import uuid
import os
import numpy as np
from werkzeug.utils import secure_filename
from keras.applications import VGG16
import pandas as pd
import numpy as np
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import h5py
from skimage.io import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization
import numpy as np
from keras import models, layers, optimizers
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

app = Flask(__name__)

#Load model
new_model=models.load_model('../vgg16cat.h5')
#extract features from model
model_extract = Model(inputs=new_model.input, outputs=new_model.get_layer('dense_32').output)
graph = tf.get_default_graph()
UPLOAD_FOLDER = '/Users/hiranya/week4ds06/class_lectures/week08-fletcher2/02/final_simple_flask/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

# @app.route("/")
# def template_test():
#     return render_template('template.html', label='', imagesource='./uploads/template.jpg')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method=='GET':
        return render_template('template.html')

    if request.method == 'POST':
       
        #start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(file_path)
            with graph.as_default():
                img_url, img_path, price, pg_url= make_prediction(file_path)

            img_url1=img_url[0]
            img_path1=img_path[0]
            price1=price[0]
            pg_url1=pg_url[0]

            img_url2=img_url[1]
            img_path2=img_path[1]
            price2=price[1]
            pg_url2=pg_url[1]

            img_url3=img_url[2]
            img_path3=img_path[2]
            price3=price[2]
            pg_url3=pg_url[2]
                 
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', image_url='./uploads/' + filename, price=1030,
                                    img_url1=img_url1,price1=price1, img_path1=img_path1,pg_url1=pg_url1,
                                    img_url2=img_url2,price2=price2, img_path2=img_path2,pg_url2=pg_url2,
                                    img_url3=img_url3,price3=price3, img_path3=img_path3,pg_url3=pg_url3)
                                    
        


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)



if __name__ == '__main__':
    app.run(debug=False)

