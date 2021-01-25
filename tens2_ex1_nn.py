import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import logging
from tensorflow import keras

################################################################
################################################################
#Dataset sa kerasa

fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training

#print(train_labels)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#plt.figure()
#plt.imshow(train_images[70])
#plt.colorbar()
#plt.grid(False)
#plt.show()

train_images = train_images / 255   #MANJE VRIJEDNOSTI DAJU BOLJE REZULTATE
test_images = test_images /255


################################################################
################################################################   
#Gradimo model

model = keras.Sequential([
							keras.layers.Flatten(input_shape = (28,28)),		#INPUT
							keras.layers.Dense(128, activation = 'relu'),		#MID   128 je hyper parametar
							keras.layers.Dense(10, activation = 'softmax')		
						])

#Kompajliramo model -> tu biramo metodu optimizacije (npr. gradient descent),
								#cost funkciju (npr. suma razlike kvadrata, i metrike koje želimo pratit)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',			#umjesto ADAM mozes: Gradient Descent, Stochastic Gradient Descent, 
              metrics=['accuracy'])								#Mini-Batch Gradient Descent,  Momentum, Nesterov Accelerated Gradient


################################################################
################################################################
#Treniramo model

model.fit(train_images, train_labels, epochs=10)

#Evaluiramo/Testiramo model

(test_loss, test_acc) = model.evaluate(test_images,  test_labels, verbose=1)    #verbose je progress bar

print('Test accuracy:', test_acc)

#Radimo predikcije

predictions = model.predict(test_images)

print(np.argmax(predictions[0]))
print(test_labels[0])

