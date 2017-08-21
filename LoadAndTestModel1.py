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

x_test = [] #test data image info
w, h = 28, 28

print('Reading CSV testing file...')
with open('input/test.csv', 'r') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV:
		if (row[0] != 'pixel0'):
			pixels = np.array(row);

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
			x_test.append(img)
print('CSV testing file read.')

x_test = np.array(x_test)

print('Loading model...')
json_file = open('Model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("Model1.h5")
print('Model loaded.')

print('Testing model...')
finalGuesses = []
predictions = loaded_model.predict(x_test)
for i in range(0, len(x_test)):
	curr = 0
	high = 0
	while curr < len(predictions[i]):
		if predictions[i][curr] > predictions[i][high]:
			high = curr
		curr += 1
	finalGuesses.append(high)
print('Model tested.')

print('Writing tested results to "results.csv"...')
with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ImageId', 'Label'])
    for i in range (1, len(finalGuesses) + 1):
    	writer.writerow([i, finalGuesses[i-1]])
print('Tested results written to "results.csv".')