import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout



model = load_model('traffic_sign.h5')
classes = { 0:'30',
            1:'40', 
            2:'50' ,
            3:'60',
            4:'70',
            5:'80'}
classes.get(0)
def test_on_img(img,classes):
    data=[]
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict(X_test)
    return classes.get(np.where((Y_pred[0]>0.5))[0][0])
