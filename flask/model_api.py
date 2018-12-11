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

#Load model
new_model=models.load_model('../vgg16cat.h5')
#extract features from model
model_extract = Model(inputs=new_model.input, outputs=new_model.get_layer('dense_32').output)

#function to convert image to np array
def get_input(img_path):
    img = imread(img_path)
    img = resize(img, (224, 224,3), preserve_range=True).astype(np.float32)
    #img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
fea=pd.read_csv('../recommender_final_flask.csv')
#print(fea.head())
# Test shoes with price greater than 300$ give the image path of file
df_price=fea.loc[fea['price']>300,:]
#get path of sample 
#sample=df_price['img_path'][9485]
sample='/Users/hiranya/week4ds06/class_lectures/week08-fletcher2/02/final_simple_flask/shoe_images/https:__www.shoes.com_vince-camuto-cashane-ankle-strap-sandal_846676_9351.jpg'
def make_prediction(sample):
    samp_mat=get_input(sample)
    samp_mat=np.expand_dims(samp_mat,axis=0)
    #predict model
    print(samp_mat.shape)
    samp_pred=model_extract.predict(samp_mat)
    print(samp_pred.shape)
    #read features of vgg16 from saved file
    fea=pd.read_csv('/Users/hiranya/week4ds06/class_lectures/week08-fletcher2/02/recommender_final_flask.csv')
    #convert df to nparray for cosine similarity calc
    search_mat=np.array(fea.iloc[:,6:])
    method='cosine-similarity'
    if method=='cosine-similarity':
        similarity=cosine_similarity(search_mat,samp_pred)
        df_sim=pd.DataFrame(similarity).sort_values(0,ascending=False).head(5)
    #list of all recommendations
    rec=fea.iloc[list(df_sim.index),0:6].values
    #print(rec)
    img_url=[]
    img_path=[]
    price=[]
    pg_url=[]
    for i in range(5):
        img_url.append(rec[i][5])
        img_path.append(rec[i][0])
        price.append(rec[i][1])
        pg_url.append(rec[i][4])
    return img_url,img_path,price,pg_url
if __name__ == '__main__':
    print(make_prediction(sample))
