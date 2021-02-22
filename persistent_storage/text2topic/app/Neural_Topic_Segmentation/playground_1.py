import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Softmax,Dot,Reshape
from tensorflow.keras import activations


max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review

# topic encdoding BiLSTM layer with attention
class self_attention(keras.layers.Layer):
    def __init__(self,sequence_length=30,hidden_state_length=128):
        super(self_attention, self).__init__()
        # create an initializer for the attention weighting variable 
        u_w_init = tf.random_normal_initializer()
        # create attentiion variable, 1 long 
        self.u_w = tf.Variable(
            initial_value=u_w_init(shape=(1,hidden_state_length*2), dtype="float32"),
            trainable=True,
        )
        self.dense_layer   = Dense(units=hidden_state_length*2,activation=activations.tanh)
        self.softmax_layer = Softmax()
        self.dot_layer1    = Dot(axes=(1))
        self.dot_layer2    = Dot(axes=(1))
        self.reshaper      = Reshape(target_shape=(hidden_state_length*2,sequence_length,))
        self.sequence_length=sequence_length
        self.hidden_state_length=hidden_state_length
        # self.input_dim = input_dim


    def call(self, h_it):
        # u_it should be (None,40,512)
        u_it=self.dense_layer(h_it)
        # a_it should be (None,1,512)
        softmax_in = self.dot_layer1([self.reshaper(u_it),self.u_w])
        a_it=self.softmax_layer(softmax_in)
        return self.dot_layer2([a_it,h_it])



# Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer in a 128-dimensional vector
x = layers.Embedding(max_features, 128)(inputs)
# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
# x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.ConvLSTM2D(
    filters=1,
    kernel_size=1,
    strides=3,
    padding="valid",
    return_sequences=True,
    go_backwards=True,
)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()



(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)


#model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
#model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))

