import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing import sequence

import os
import numpy as np

##################################################################
##################################################################
# LOAD DATA 

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
	return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

def int_to_text(ints):
	try:
		ints = ints.numpy()
	except:
		pass
	return ''.join(idx2char[ints])

# stvaranje primjera, ulazi su "rečenice" duljine 250 znakova
# izlaz isti samo pomaknut za 1 znak -> predviđamo 1 po 1 znak

seq_length = 100  # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# batches of length 100
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):  # for the example: hello
	input_text = chunk[:-1]  # hell
	target_text = chunk[1:]  # ello
	
	return input_text, target_text  # hell, ello

dataset = sequences.map(split_input_target)  # we use map to apply the above function to every entry

#TRAINING BATCHES
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

##################################################################
##################################################################
# BUILD MODEL


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

# stvaramo svoju loss funkciju jer je izlaz 3D polje - 64 ulaza, po 100 znakova po 65 klasa. za svaki ulaz radi se predikcija u svakom trenutku,
# pa biramo po normalnoj distribuciji izlaz za svaki trenutak (najveći od trenutaka najčešće (za svaki trenutak je 65 izlaza))
def loss(labels, logits):
	return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# kompajliramo model
model.compile(optimizer='adam', loss=loss)


# SAVING CHECKPOINTS
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_prefix,
	save_weights_only=True)



# TRAINING 
history = model.fit(data, epochs=50, callbacks=[checkpoint_callback])


##################################################################
##################################################################
# PREPARING FOR PREDICTIONS

# BATCH SIZE = 1, 1 text za predikciju, RBUILDAMO I ONDA STAVIMO SAVEANE WEIGHTOVE I BIAS
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# load from any checkpoint
#checkpoint_num = 10
#model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
#model.build(tf.TensorShape([1, None]))


# MAKING PREDICTIONS

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
	num_generate = 800

  # Converting our start string to numbers (vectorizing)
	input_eval = [char2idx[s] for s in start_string]
	input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
	text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
	temperature = 1.0

  # Here batch size == 1
	model.reset_states()
	
	for i in range(num_generate):
		predictions = model(input_eval)
      # remove the batch dimension
    
		predictions = tf.squeeze(predictions, 0)

    	# using a categorical distribution to predict the character returned by the model
		predictions = predictions / temperature
		predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
		input_eval = tf.expand_dims([predicted_id], 0)

		text_generated.append(idx2char[predicted_id])

	return (start_string + ''.join(text_generated))

inp = input("Type a starting string: ")
print(generate_text(model, inp))


# MORE ON LSTMs “Understanding LSTM Networks.” Understanding LSTM Networks -- Colah's Blog, https://colah.github.io/posts/2015-08-Understanding-LSTMs/.
