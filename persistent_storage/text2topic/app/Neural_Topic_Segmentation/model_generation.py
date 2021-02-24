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
from tensorflow.keras.layers import LSTM,Embedding,Dense,Multiply,Bidirectional,Softmax,Dot,Attention,MaxPooling1D,Reshape,Add,Concatenate,ConvLSTM2D,Subtract
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
class se_conv_BiLSTM(keras.layers.Layer):
    def __init__(self,
        sentence_sequence_length = 10,
        max_sentence_length = 40, 
        LSTM_hidden_state_length = 256, 
        embedding_length = 300,layer_name=""):
        super(se_conv_BiLSTM, self).__init__()
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

# sentence encoding comparison self attention BiLSTM layer
class sec_conv_BiLSTM(keras.layers.Layer):
    def __init__(self,
        window_length = 10,
        sentence_encoding_length = 1024):
        super(sec_conv_BiLSTM, self).__init__()

        self.x_forward_1 = ConvLSTM2D(
            filters = sentence_encoding_length,
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
        self.x_backward_1 = ConvLSTM2D(
            filters = sentence_encoding_length,
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
        self.sub=Subtract()
        self.reshaper_0 = Reshape(target_shape=(1,window_length,1,sentence_encoding_length))
        self.reshaper_1 = Reshape(target_shape=(window_length,1,sentence_encoding_length))
        self.reshaper_2 = Reshape(target_shape=(window_length,sentence_encoding_length,1))
        self.reshaper_3 = Reshape(target_shape=(window_length,window_length,sentence_encoding_length))
        self.reshaper_4 = Reshape(target_shape=(window_length,sentence_encoding_length))
        self.concat     = Concatenate(axis=3)
        self.multiply   = Multiply()
        self.softmax    = Softmax(axis=2)# ie the second of the 10 dims (1,10,[10],1024)


        self.x_forward_0 = ConvLSTM2D(
            filters = sentence_encoding_length,
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
        self.x_backward_0 = ConvLSTM2D(
            filters = sentence_encoding_length,
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

        self.dot = Dot(axes=(1,1))

        initializer = tf.keras.initializers.Ones()

        self.ones_a = tf.Variable(initial_value=initializer((window_length,1,)),
                               trainable=True)
        self.ones_b = tf.Variable(initial_value=initializer((1,window_length,)),
                               trainable=True)

        self.dense = Dense(units=1024,activation=activations.softmax)



    def call(self, input_t):
        x          = self.reshaper_0(input_t)
        x_forward  = self.x_forward_0(x)
        x_backward = self.x_backward_0(x)
        h          = self.sub([x_forward,x_backward])
        x_a        = self.reshaper_1(h)
        x_b        = self.reshaper_2(h)
        x_a        = linalg.matmul(self.ones_a,x_a)
        x_b        = self.reshaper_3(linalg.matmul(x_b,self.ones_b))
        x_c        = self.multiply([x_a,x_b])
        x_d        = self.concat([x_a,x_b,x_c])
        e          = self.dense(x_d)
        a          = self.softmax(e) # sums on second 10 dim
        h_flat     = self.reshaper_4(h) # 1,10,1,1024

        # this multiplicaiton is to sum together accross the channels
        # x          = linalg.matmul(a,h_flat,transpose_a=True,transpose_b=True)    # doesn't run          # doesn't run
        # x          = linalg.matmul(a,h_flat,transpose_a=False,transpose_b=True)  # outputs (1,10,10,1)   # outputs (1,10,10,10)
        # x          = linalg.matmul(a,h_flat,transpose_a=True,transpose_b=False)  # doesn't run           # outputs (1,10,1024,1024)
        x          = linalg.matmul(a,h_flat,transpose_a=False,transpose_b=False) # doesn't run



        # x          = self.reshaper_5(x)       
        # x          = self.dot([a,h_flat])


        # x          = self.reshaper_0(input_t)
        # x_forward  = self.x_forward_0(x)
        # x_backward = self.x_backward_0(x)
        # h          = self.sub([x_forward,x_backward])

        return x


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
        self.reshaper1 = Reshape(target_shape=(sentence_sequence_length,max_sentence_length,LSTM_hidden_state_length))
        self.reshaper2 = Reshape(target_shape=(sentence_sequence_length,LSTM_hidden_state_length,max_sentence_length))
        self.reshaper3 = Reshape(target_shape=(sentence_sequence_length,LSTM_hidden_state_length))
        # self.reshaper1 = Reshape(target_shape=(sentence_sequence_length,max_sentence_length,LSTM_hidden_state_length,1))
        # self.reshaper2 = Reshape(target_shape=(sentence_sequence_length,max_sentence_length,1,LSTM_hidden_state_length))
        self.softmax   = Softmax()


        initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)

        self.u_w = tf.Variable(initial_value=initializer((LSTM_hidden_state_length,1,)),
                               trainable=True)



    def call(self, x):
        temp = self.reshaper0(x)
        keys    = self.dense0(temp)
        keys    = self.reshaper1(keys)
        # queries = self.reshaper2(queries)
        x       = linalg.matmul(keys,self.u_w)
        x       = self.softmax(x)
        temp    = self.reshaper2(temp)
        x       = linalg.matmul(temp,x)
        x       = self.reshaper3(x)

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

    inputs = keras.Input(shape=(sentence_sequence_length,max_sentence_length,1,embedding_length), batch_size=batch_size,dtype="float32",name="Embedding Input Layer")
    
    x = se_conv_BiLSTM(
        sentence_sequence_length,
        max_sentence_length,
        LSTM_hidden_state_length,
        embedding_length)(inputs)
    x = se_conv_BiLSTM(
        sentence_sequence_length,
        max_sentence_length,
        LSTM_hidden_state_length,
        embedding_length)(x)
    output = custom_SAT(sentence_sequence_length,
        max_sentence_length,
        LSTM_hidden_state_length,
        embedding_length)(x)



    return inputs,output

# This function calls build_Att_BiLSTM to build the global sentence encoder
# output can be "softmax" or "attention"
def build_sentence_encoder(BiLSTM_outputs,sentence_sequence_length,batch_size,bert_embedding_length=768,output="softmax"):
    bert_input = keras.Input(shape=(sentence_sequence_length,bert_embedding_length),batch_size=batch_size, dtype="float32", name="BERT Input Layer")
    x = Concatenate()([bert_input,BiLSTM_outputs])

    output = sec_conv_BiLSTM(
        window_length = 10,
        sentence_encoding_length = 1024)(x)


    return bert_input,output



if __name__=="__main__":
    print("Running from {}".format(__file__))
    sentence_sequence_length = 10
    max_sentence_length = 40
    LSTM_hidden_state_length = 256
    embedding_length = 300
    batch_size=1
    
    raw_inputs,outputs=build_conv_Att_BiLSTM(
        sentence_sequence_length,
        max_sentence_length,
        LSTM_hidden_state_length,
        embedding_length,
        batch_size)

    bert_inputs,outputs = build_sentence_encoder(outputs,sentence_sequence_length,batch_size)




    model = keras.Model([raw_inputs,bert_inputs], outputs)
    model.summary(line_length=160)
    # plot_model(model,'model.png')



def instantiate_bert_client():
    pass

# Assembles label prediction network
def build_label_prediction_network():
    pass

