#Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()


##################################################################
##################################################################
# LOAD DATA 

# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels

# display 2 images from the dataset
#for image, label in raw_train.take(5):
#  plt.figure()
#  plt.imshow(image)
#  plt.title(get_label_name(label))

##################################################################
# PREPROCESS DATA AND BATCH IT

IMG_SIZE = 160 # All images will be resized to 160x160

# returns an image that is reshaped to IMG_SIZE
def format_example(image, label):
	image = tf.cast(image, tf.float32)
	image = (image/127.5) - 1
	image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

	return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


##################################################################
##################################################################
# MODEL


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


base_model.trainable = False # freeze base model

# add our classifier on top

global_average_layer = keras.layers.GlobalAveragePooling2D()  # POOLS ALL 5X5s, returns 1280 vector, per feature

prediction_layer = keras.layers.Dense(1) # PREDVIĐAMO ZA PSE/MAČKE 1 NEURON JE DOVOLJAN

model = keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])


##################################################################
##################################################################
# TRENIRAMO MODEL


base_learning_rate = 0.0001
model.compile(optimizer= keras.optimizers.RMSprop(lr=base_learning_rate),
              loss= keras.losses.BinaryCrossentropy(from_logits=True),     # BINARY JER SU 2 KLASE
              metrics=['accuracy'])

initial_epochs = 1
validation_steps=20

# We can evaluate the model right now to see how it does before training it on our new images, WITHOUT LEARNING
loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)


# Now we can train it on our images
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)


##################################################################
##################################################################
# SAVE THE MODEL FOR FUTURE USE

#model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future
#new_model = keras.models.load_model('dogs_vs_cats.h5')

# TF OBJECT DETECTION: https://github.com/tensorflow/models/tree/master/research/object_detection
# python facial recognition model

