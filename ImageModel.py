# Author: Omkar
# Image Classification model using CNN
# Code referred from Week 8 assignments


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
import os

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True


#Load Data

rgbstart = time.time()
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Rgb Start Time =", current_time)

from PIL import Image

Train_source = 'Train Set'
Test_source = 'Test Set'
Label_List=["food", "sports", "travel", "technology", "science"]

Train_files = os.listdir(Train_source)
Test_files = os.listdir(Test_source)
x_train=[]
x_test=[]
y_train=[]
y_test=[]

Dimension=32    #64

print("Processing Train Files")
for f in Train_files:
    im = Image.open(Train_source+'/'+f)
    new_im = im.resize((Dimension, Dimension))
    rgb = np.array(new_im.convert('RGB'))
    x_train.append(rgb)
    label=[]
    label.append(Label_List.index(f.split('_')[0]))
    label=np.array(label)
    y_train.append(label)


print("Processing Test Files")
for f in Test_files:
    im = Image.open(Test_source+'/'+f)
    new_im = im.resize((Dimension, Dimension))
    rgb = np.array(new_im.convert('RGB'))
    x_test.append(rgb)
    label=[]
    label.append(Label_List.index(f.split('_')[0]))
    label=np.array(label)
    y_test.append(label)

#Convert Lists into np arrays
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)


print("X train shape")
print(x_train.shape)

print("X test shape")
print(x_test.shape)

print("Y train shape")
print(y_train.shape)

print("Y test shape")
print(y_test.shape)


# Model / data parameters
num_classes = 5
input_shape = (Dimension, Dimension, 3)

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print("orig x_train shape:", x_train.shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, test_size=0.2, random_state=42)


print("Rgb End Time =", datetime.now().strftime("%H:%M:%S"))
print("Total Time Taken (secs): ", (time.time() - rgbstart))


start = time.time()
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)


model = keras.Sequential()
# model.add(Conv2D(4, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))#strides=(2,2)
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(8, (3,3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3,3), padding='same',input_shape=x_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(0.0001))) #0.0001,0.01, 0.1, 0,1
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()
exit()
batch_size = 128
epochs = 20
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val,y_val) ) #validation_split=0.2
model.save("subreddit.model")

print("End Time =", datetime.now().strftime("%H:%M:%S"))
print("Total Time Taken (secs): ", (time.time() - start))

plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss'); plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.tight_layout()
plt.show()


preds = model.predict(x_train)
y_pred = np.argmax(preds, axis=1)
y_train1 = np.argmax(y_train, axis=1)
print(classification_report(y_train1, y_pred))
print(confusion_matrix(y_train1,y_pred))

preds = model.predict(x_test)
y_pred = np.argmax(preds, axis=1)
y_test1 = np.argmax(y_test, axis=1)
print(classification_report(y_test1, y_pred))
print(confusion_matrix(y_test1,y_pred))