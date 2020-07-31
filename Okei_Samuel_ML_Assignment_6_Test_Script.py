import numpy as np
import matplotlib.pyplot as pl
import random
from math import exp
import time
import os
import glob
import cv2
import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras import backend as K


start = time.time()
X_test = []

content = input("Please enter the directory containing the extracted celegans folder: ")

img_dir = content+"\celegans\\0\\test"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    X_test.append(np.array([img, 0]))


img_dir = content+"\celegans\\1\\test"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    X_test.append(np.array([img, 1]))


X_test = np.array(X_test)

np.random.seed(57)
np.random.shuffle(X_test)

x_test = X_test[:, 0]
y_test = X_test[:, 1:2]

test_size = y_test.shape[0]
x__test = np.zeros([test_size, 101,101])

for i in range(len(x_test)):
    x__test[i] = x_test[i]
    
img_rows = 101
img_cols = 101    
x_test = x__test

mean = 159.5736681074048
std = 30.022396270385883

x_test = (x_test-mean)/std
x_test = x_test.reshape(x_test.shape[0],101,101,1)

y_test = keras.utils.to_categorical(y_test, 2)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model.h5")

start = time.time()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
score = model.evaluate(x_test, y_test, verbose=0)

end = time.time()
print('Test time:', end - start)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



