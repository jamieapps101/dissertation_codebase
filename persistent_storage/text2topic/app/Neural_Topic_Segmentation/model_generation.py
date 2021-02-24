# This script puts togeather a model based on "Improving Context Modeling in Neural Topic Segmentation"

# Obvs:
import tensorflow as tf

# Required for BERT
import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.executing_eagerly()

# import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()


from tensorflow import keras
from tensorflow.keras.layers import LSTM,Embedding,Dense,Multiply,Bidirectional,Softmax,Dot,Attention,MaxPooling1D,Reshape,Add,Concatenate,ConvLSTM2D
from tensorflow.keras import activations
from tensorflow import linalg
from tensorflow.keras.utils import plot_model

# top level function, to be used by external scripts
# This function calls all remaining functions in this file and
# returns a fully build model
def build_model():
    # Call all required functions
    # compile the model
    # return it
    pass

# combines Att BiLSTM and bert models
def build_sentence_encoding_network():
    pass

# self attention network to place ontop BiLSTM
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

# sentence encdoding BiLSTM layer
class conv_BiLSTM(keras.layers.Layer):
    def __init__(self,
        sentence_sequence_length = 10,
        max_sentence_length = 40, 
        LSTM_hidden_state_length = 256, 
        embedding_length = 300,layer_name=""):
        super(conv_BiLSTM, self).__init__()
        self.x_forward = ConvLSTM2D(
            filters = 256,
            kernel_size=(1,1),
            strides=(1,1),
            padding="valid",
            data_format="channels_last",
            # recurrent_activation=activations.tanh, # check this
            return_sequences=True,
            go_backwards=False,
            stateful=True,
            # dropout=0.2,
            # recurrent_dropout=0.2
        )
        self.x_backward = ConvLSTM2D(
            filters = 256,
            kernel_size=(1,1),
            strides=(1,1),
            padding="valid",
            data_format="channels_last",
            recurrent_activation=activations.tanh, # check this
            return_sequences=True,
            go_backwards=True,
            stateful=True,
            # dropout=0.2,
            # recurrent_dropout=0.2
        )
        self.add=Add()


    def call(self, input_t):
        return self.add([self.x_forward(input_t),self.x_backward(input_t)])


class custom_SAT(keras.layers.Layer):
    def __init__(self,
        sentence_sequence_length = 10,
        max_sentence_length = 40, 
        LSTM_hidden_state_length = 256, 
        embedding_length = 300):
        super(custom_SAT, self).__init__()

        self.reshaper0 = Reshape(target_shape=(sentence_sequence_length,max_sentence_length,LSTM_hidden_state_length))
        self.dense0    = Dense(units=LSTM_hidden_state_length,activation=activations.tanh)
        self.dense1    = Dense(units=LSTM_hidden_state_length,activation=activations.tanh)
        self.reshaper1 = Reshape(target_shape=(sentence_sequence_length,max_sentence_length,LSTM_hidden_state_length,1))
        self.reshaper2 = Reshape(target_shape=(sentence_sequence_length,max_sentence_length,1,LSTM_hidden_state_length))
        self.softmax   = Softmax()


    def call(self, x):
        temp = self.reshaper0(x)
        keys    = self.dense0(temp)
        queries = self.dense1(x)
        # do some weird reshaping here to allow matrix multiplication
        keys    = self.reshaper1(keys)
        queries = self.reshaper2(queries)
        x       = linalg.matmul(keys,queries)
        x       = self.softmax(x)
        temp    = self.reshaper1(temp)
        x       = linalg.matmul(x,temp)
        x       = self.reshaper0(x)

        return x
        



# max_sentence_length makes building the network simpler
# input sentences shorter than this will need padding    
def build_conv_Att_BiLSTM(
    sentence_sequence_length = 10,
    max_sentence_length = 40, 
    LSTM_hidden_state_length = 256, 
    embedding_length = 300, 
    batch_size=1,
    self_attention_enabled=True):

    # inputs = keras.Input(shape=(sentence_sequence_length,max_sentence_length,embedding_length,1,), dtype="float32")
    inputs = keras.Input(shape=(sentence_sequence_length,max_sentence_length,1,embedding_length), batch_size=batch_size,dtype="float32",name="Embedding Input Layer")
    
    x = conv_BiLSTM(
        sentence_sequence_length,
        max_sentence_length,
        LSTM_hidden_state_length,
        embedding_length)(inputs)
    x = conv_BiLSTM(
        sentence_sequence_length,
        max_sentence_length,
        LSTM_hidden_state_length,
        embedding_length)(x)
    
    # temp = Reshape(target_shape=(sentence_sequence_length,max_sentence_length,LSTM_hidden_state_length))(x)

    # keys = Dense(units=LSTM_hidden_state_length,activation=activations.tanh)(temp)
    # queries = Dense(units=LSTM_hidden_state_length,activation=activations.tanh)(x)
    # # do some weird reshaping here to allow matrix multiplication
    # keys    = Reshape(target_shape=(sentence_sequence_length,max_sentence_length,LSTM_hidden_state_length,1))(keys)
    # queries = Reshape(target_shape=(sentence_sequence_length,max_sentence_length,1,LSTM_hidden_state_length))(queries)
    # x=linalg.matmul(keys,queries)
    # x=Softmax()(x)
    # temp    = Reshape(target_shape=(sentence_sequence_length,max_sentence_length,LSTM_hidden_state_length,1))(temp)
    # x=linalg.matmul(x,temp)
    # x = Reshape(target_shape=(sentence_sequence_length,max_sentence_length,LSTM_hidden_state_length))(x)

    x = custom_SAT(sentence_sequence_length,
        max_sentence_length,
        LSTM_hidden_state_length,
        embedding_length)(x)

    # else:
    #     # else use the original paper'ss max pooling
    #     x = Reshape(target_shape=(max_sentence_length,LSTM_hidden_state_length*2))(x)
    #     outputs = MaxPooling1D(pool_size=128, strides=None, padding="same", data_format="channels_last")(x)

    model = keras.Model(inputs, x)
    return model
    # return inputs, x

# This function calls build_Att_BiLSTM to build the global sentence encoder
def build_sentence_encoder(bert_embedding_length=768):
    BiLSTM_inputs,BiLSTM_outputs = build_conv_Att_BiLSTM(
        max_sentence_length = 40, 
        LSTM_hidden_state_length = 256, 
        embedding_length = 300,
        self_attention_enabled=True)
    
    bert_input = keras.Input(shape=(bert_embedding_length,), dtype="float32")

    output = Concatenate()([bert_input,BiLSTM_outputs])

    return [BiLSTM_inputs,bert_input],output



if __name__=="__main__":
    print("Running from {}".format(__file__))
    
    model=build_conv_Att_BiLSTM()

    # inputs,output = build_sentence_encoder()
    # model = keras.Model(inputs, output)
    model.summary(line_length=160)
    # plot_model(model,'model.png')



def instantiate_bert_client():
    pass

# Assembles label prediction network
def build_label_prediction_network():
    pass

