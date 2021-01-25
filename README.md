# tensorflow_exercise
Exercies in tensorflow to learn the framework

Ecercises in tensorflow and keras to learn the basics of the framework.

Includes:
1. Linear Regression to determine survival aboard titanic -> tf.estimator.LinearClassifier

2. Classification of different wheat crops (3 categories) -> tf.estimator.DNNClassifier (DNN - Dense Neural Network)

3. Classification of clothing items from images (10 categories) -> keras.Sequential (Flatten + 2 DNN layers)

4. Classification of general objects or animals from images (Convolutional NN) -> (layers.Conv2D(), layers.MaxPooling2D())

5. Differentiating cats and dogs with a PRETRAINED (MobileNetV2) CNN -> (keras.applications.MobileNetV2(), keras.layers.GlobalAveragePooling2D())

6. Sentiment analysis of movie reviews using a Recurrent Neural network -> (tf.keras.layers.Embedding(), tf.keras.layers.LSTM()) 

7. Generating a text document using another document with RNN -> (data splitting and encoding, recurrent_initializer='glorot_uniform')

6. Q learning simple example -> Open AI GYM
