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

X_train = []
X_test = []

initialtime = time.time()

content = input("Please enter the directory containing the extracted celegans folder: ")

img_dir = content+"\celegans\\0\\training"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    X_train.append(np.array([img, 0]))


img_dir = content+"\celegans\\1\\training"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    X_train.append(np.array([img, 1]))


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


X_train = np.array(X_train)
X_test = np.array(X_test)

np.random.seed(57)

np.random.shuffle(X_train)
np.random.shuffle(X_test)

x_train = X_train[:, 0]
y_train = X_train[:, 1:2]

x_test = X_test[:, 0]
y_test = X_test[:, 1:2]

train_size = y_train.shape[0]
test_size = y_test.shape[0]

x__train = np.zeros([train_size, 101,101])
x__test = np.zeros([test_size, 101,101])


for i in range(len(x_train)):
    x__train[i] = x_train[i]

for i in range(len(x_test)):
    x__test[i] = x_test[i]

mean = 0
std = 0

img_rows = 101
img_cols = 101
num_classes = 2
   
x_train = x__train
x_test = x__test

mean = np.mean(x_train)
std = np.std(x_train)

x_train = (x_train-mean)/std
x_train = x_train.reshape(x_train.shape[0],101,101,1)

x_test = (x_test-mean)/std
x_test = x_test.reshape(x_test.shape[0],101,101,1)
   
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
encoder = OneHotEncoder()

y_train = encoder.fit_transform(y_train.reshape((-1,1)))
y_train = y_train.toarray()

y_test = encoder.transform(y_test.reshape((-1,1)))
y_test = y_test.toarray()

#create model

model = Sequential() #add model layers
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(64, kernel_size=5, activation='relu'))
model.add(MaxPooling2D(pool_size=(10,10)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))


#compile model using accuracy to measure model performance

model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=6)
print(time.time()-initialtime)

score_ = model.evaluate(x_train, y_train)
score = model.evaluate(x_test, y_test)

print('Train loss:', score_[0])
print('Train accuracy:', score_[1])

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
