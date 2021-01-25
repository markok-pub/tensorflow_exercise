import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import logging


################################################################################
################################################################################
#DATASET

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of Pandas) to grab our datasets and read them into a pandas dataframe

y_train = train.pop('Species')
y_test  = test.pop('Species')

print(train.shape)

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


################################################################################
################################################################################
#INPUT

def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

################################################################################
################################################################################
#STVARANJE MODELA


# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
	feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
	n_classes=3)


################################################################################
#TRENIRANJE MODELA

logging.getLogger().setLevel(logging.INFO)  # za printanje progressa

classifier.train(
	input_fn = lambda: input_fn(train, y_train, training = True),
		steps = 5000) 		#5000 puta pogledaj red dataseta, ovo je kao epoha


################################################################################
#EVALUACIJA MODELA

result = classifier.evaluate(input_fn = lambda: input_fn(test, y_test, training = False))

print(result['accuracy'])

################################################################################
#  PREDICTION: //// 1. napravis dictionary pred[feature] = [value] ///// 2. napravis: input_fn : from_tensor_slices(dict(features)).batch(256) ////// 3. predas pred kao features
#			   //// 4. classifier.predict()




