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
from tensorflow.keras.layers import LSTM,Embedding,Dense,Bidirectional,Softmax,Dot,Attention,MaxPooling1D,Reshape,Add,Concatenate
from tensorflow.keras import activations
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


# max_sentence_length makes building the network simpler
# input sentences shorter than this will need padding    
def build_Att_BiLSTM(max_sentence_length = 40, LSTM_hidden_state_length = 256, embedding_length = 300, self_attention_enabled=True):
    comb_opts=['sum', 'mul', 'concat', 'ave', None]
    inputs = keras.Input(shape=(max_sentence_length,embedding_length,), dtype="float32")
    
    x = Bidirectional(LSTM(LSTM_hidden_state_length, activation="tanh",dropout=0.2,recurrent_dropout=0.2, return_sequences=True),merge_mode=comb_opts[2])(inputs)
    x = Bidirectional(LSTM(LSTM_hidden_state_length, activation="tanh",dropout=0.2,recurrent_dropout=0.2, return_sequences=True),merge_mode=comb_opts[2])(x)
    
    outputs = None
    if self_attention_enabled:
        # either use the ewer paper's self attention
        outputs = self_attention(max_sentence_length,LSTM_hidden_state_length)(x)
    else:
        # else use the original paper'ss max pooling
        x = Reshape(target_shape=(max_sentence_length,LSTM_hidden_state_length*2))(x)
        outputs = MaxPooling1D(pool_size=128, strides=None, padding="same", data_format="channels_last")(x)

    # model = keras.Model(inputs, outputs)
    # return model
    return inputs, outputs

# This function calls build_Att_BiLSTM to build the global sentence encoder
def build_sentence_encoder(bert_embedding_length=768):
    BiLSTM_inputs,BiLSTM_outputs = build_Att_BiLSTM(
        max_sentence_length = 40, 
        LSTM_hidden_state_length = 256, 
        embedding_length = 300,
        self_attention_enabled=True)
    
    bert_input = keras.Input(shape=(bert_embedding_length,), dtype="float32")

    output = Concatenate()([bert_input,BiLSTM_outputs])

    return [BiLSTM_inputs,bert_input],output

class auxilary_task_module(keras.layers.Layer):
    def __init__(self, window_width=4):
        super(auxilary_task_module, self).__init__()


    def call(self):
        pass

# This class acts as a single layer, but contains the full sentence
# level comparison network
class sentence_BiLSTM_self_att(keras.layers.Layer):
    def __init__(self,sentence_sequence_length=5,bilstm_hidden_states=256):
        super(sentence_BiLSTM_self_att, self).__init__()
        # create first bi-lstm layer
        self.BiLSTM1 = Bidirectional(
            LSTM(
                bilstm_hidden_states, 
                activation="tanh",
                dropout=0.2,
                recurrent_dropout=0.2, 
                return_sequences=True),
            merge_mode="sum")



    def call(self, h_it):
        pass


if __name__=="__main__":
    print("Running from {}".format(__file__))
    
    # model=build_Att_BiLSTM()

    inputs,output = build_sentence_encoder()
    model = keras.Model(inputs, output)
    model.summary()
    plot_model(
        model, 
        to_file='model.png', 
        show_shapes=False, 
        show_dtype=True,
        show_layer_names=True, 
        rankdir='TB', 
        expand_nested=True, 
        dpi=96
)



def instantiate_bert_client():
    pass

# Assembles label prediction network
def build_label_prediction_network():
    pass

