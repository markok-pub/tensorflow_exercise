import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##################################################################
##################################################################
# LOAD DATA

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# Let's look at a one image
IMG_INDEX = 7  # change this to look at other images

plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()


##################################################################
##################################################################
# CREATE MODEL

model = models.Sequential()

# dodajemo konvolucijske slojeve

model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32, 32, 3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

# dodajemo Dense slojeve

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))


##################################################################
# treniramo model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   # SVE ISTO KAO U DNN-u, FUNKCIJA LOADANA IZ KERASA 'sparse_categorical_crossentropy'
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))        # TRENING I EVALUACIJA ODJEDNOM 


##################################################################

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)  # EVALUACIJA JOS JEDNOM, PROVJERI KAJ RADI HISTORY 
print(test_acc)																# HISTORY: acc = history.history['accuracy'] umjesto evaluate




