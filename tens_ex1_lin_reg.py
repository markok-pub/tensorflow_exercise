from __future__ import absolute_import, division, unicode_literals, print_function

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow.compat.v2.feature_colums as fc
from IPython.display import clear_output
from six.moves import urllib



##########################################################################################
##########################################################################################
#ČITANJE PODATAKA

dtrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  
deval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
#print(deval.head())
y_train = dtrain.pop('survived')    #POP MICE STUPAC IZ dtrain I STAVLJA GA U y_train
y_eval = deval.pop('survived')		# .loc[i] locira red i
									# .describe() ispisuje statistike o datasetu, .head() ispisuje prvih 5 redova

#pd.concat([dtrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
#plt.show()  ---> POKAZUJE PLOT

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
	vocabulary = dtrain[feature_name].unique()  # gets a list of all unique values from given feature column
	feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
	feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


##########################################################################################
##########################################################################################
#ULAZNA FUNKCIJA   

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):    #OVO RJESI LAMBOM INACE, TRENING PRIMA FUNKCIJU
	
	def input_function():  # inner function, this will be returned
		ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its labe
		if shuffle:
			ds = ds.shuffle(1000)  # randomize order of data
		ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
		return ds  # return a batch of the dataset

	return input_function  # return a function object for use

train_input_fn = make_input_fn(dtrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(deval, y_eval, num_epochs=1, shuffle=False)

##########################################################################################
##########################################################################################
#STVARANJE MODELA


linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)


#TRENIRANJE MODELA

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model

##########################################################################################
##########################################################################################
#KORIŠTENJE MODELA

rez = list(linear_est.predict(eval_input_fn))

wt_names = linear_est.get_variable_names()
wt_vals = [linear_est.get_variable_value(name) for name in wt_names]

for i in range(1,10):
	print(str(wt_names[i])+" "+str(wt_vals[i]))

plt.show()

