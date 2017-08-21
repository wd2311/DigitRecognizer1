from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_json
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras import backend as K
import numpy as np
import pandas as pd
import csv
import os
import theano

K.set_image_dim_ordering('th'); print()
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
theano.config.optimizer = "None"

x = [] #training data image info
y = [] #corresponding labels

w, h = 28, 28

print('Reading CSV training file...')
with open('input/train.csv', 'r') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV:
		if (row[0] != 'label'):
			pixels = np.array(row[1:785]);

			data = np.zeros((h, w, 3), dtype=np.uint8)
			for i in range (0, (h*w)):
				xPos = int(i/w)
				yPos = i - w*int(i/w)
				pixVal = int(pixels[i])
				data[xPos, yPos] = [pixVal, pixVal, pixVal]
			img = Image.fromarray(data, 'RGB')
			img = img_to_array(img) / 255
			img = img.transpose(2, 0, 1)
			img = img.reshape(3, h, w)


			x.append(img)
			y.append(int(row[0]))
print('CSV training file read.')

x = np.array(x)
y = np.array(y)

batch_size = 32
nb_classes = 10 #10 digits
nb_epoch = 3
nb_filters = 32
nb_pool = 2
nb_conv = 3

uniques, id_train = np.unique(y, return_inverse=True)
y_train = np_utils.to_categorical(id_train, nb_classes)

print('Creating model...')
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', input_shape=x.shape[1:]))
model.add(Activation('relu'));
model.add(Convolution2D(nb_filters, nb_conv, nb_conv));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)));
model.add(Dropout(0.5));
model.add(Flatten());
model.add(Dense(128));
model.add(Dropout(0.5));
model.add(Dense(nb_classes));
model.add(Activation('softmax'));
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
print('Model created.')

nb_epoch = 5;
batch_size = 5;

print('Fitting model...')
model.fit(x, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
print('Fitting complete.')

print('Saving model...')
model_json = model.to_json()
with open("Model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("Model.h5")
print('Model has been saved.')