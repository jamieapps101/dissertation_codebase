# This script puts togeather a model based on "Improving Context Modeling in Neural Topic Segmentation"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# Obvs:
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.executing_eagerly()

# import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()


from tensorflow import keras
from tensorflow.keras.layers import LSTM,Embedding,Dense,Multiply,Bidirectional,Softmax,Dot,Attention,MaxPooling1D,Reshape,Add,Concatenate,ConvLSTM2D,Subtract,RepeatVector
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
            filters = 1,
            kernel_size=(1,1),
            strides=(1,1),
            padding="valid",
            data_format="channels_last",
            recurrent_activation=activations.tanh, # check this
            return_sequences=True,
            go_backwards=False,
            stateful=True,
            # dropout=0.2,
            # recurrent_dropout=0.2
        )
        self.x_backward_1 = ConvLSTM2D(
            filters = 1,
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
        self.reshaper_4 = Reshape(target_shape=(window_length*sentence_encoding_length,))
        self.reshaper_5 = Reshape(target_shape=(window_length,window_length,sentence_encoding_length))
        self.reshaper_6 = Reshape(target_shape=(window_length,sentence_encoding_length))
        self.reshaper_7 = Reshape(target_shape=(1,window_length,1,sentence_encoding_length*2))
        self.reshaper_8 = Reshape(target_shape=(1,window_length))

        self.concat_0   = Concatenate(axis=3)
        self.concat_1   = Concatenate(axis=2)
        self.multiply   = Multiply()
        self.softmax    = Softmax(axis=2)# ie the second of the 10 dims (1,10,[10],1024)
        self.softmax_1    = Softmax(axis=1)# (1,[10])

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
        self.ones_a = tf.Variable(initial_value=initializer((window_length,1,)),trainable=True)
        self.ones_b = tf.Variable(initial_value=initializer((1,window_length,)),trainable=True)
        self.dense = Dense(units=sentence_encoding_length,activation=activations.softmax)
        self.repeater0 = RepeatVector(window_length)

    def call(self, input_t): # (1,10,1024)
        # 1st bilstm
        x          = self.reshaper_0(input_t)
        x_forward  = self.x_forward_0(x)
        x_backward = self.x_backward_0(x)
        h          = self.sub([x_forward,x_backward])
        h_reduce   = self.reshaper_6(h)

        # concat data
        x_a        = self.reshaper_1(h)
        x_b        = self.reshaper_2(h)
        x_ab        = linalg.matmul(self.ones_a,x_a)
        x_bb        = self.reshaper_3(linalg.matmul(x_b,self.ones_b))
        x_c        = self.multiply([x_ab,x_bb])
        x_d        = self.concat_0([x_ab,x_bb,x_c])

        # dense fully connected layer with softmax
        e          = self.dense(x_d)
        a          = self.softmax(e) # norms accross second 10 dim

        # self attention mechanism (though really h_flat should go through a dense layer too)
        h_flat     = self.reshaper_4(h) # (1,1,10,1,1024)->(1,10*1024)
        # a          = self.reshaper_5(a) # 1,10,10,1024

        # NOTE - tensor repeating occurs here as the multiply option required
        # for the operation in the original paper was not availiable, so this was done
        # instead. Here the original sentence embedding "h_flat" (1,10,1024) is extruded into
        # a square cross section (1,[10],10,1024), where [] indicates the repeated dimension.
        # The multiply operation is then used with "a" (1,10,10,1024), which does element wise 
        # multiplication to form c_square (1,[10],10,1024), which is then summed along the []
        # axis
        h_flat_stack = self.repeater0(h_flat) # (1,10*1024)->(1,10,10*1024) 
        h_square = self.reshaper_5(h_flat_stack) # (1,10,10*1024) -> (1,10,10,1024) 
        c_square = self.multiply([a,h_square])
        c = tf.math.reduce_sum(c_square,axis=1) # (1,[10],10,1024) -> (1,10,1024) 

        # concat with original sentence hidden states
        h_2       = self.concat_1([c,h_reduce])

        #second BiLSTM layer
        x          = self.reshaper_7(h_2)
        x_forward  = self.x_forward_1(x)
        x_backward = self.x_backward_1(x)
        h2          = self.sub([x_forward,x_backward])
        h2_reduce  = self.reshaper_8(h2)

        out        = self.softmax_1(h2_reduce)

        return out

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
def build_sentence_encoder(BiLSTM_outputs,sentence_sequence_length,batch_size,bert_embedding_length=1024,output="softmax"):
    bert_input = keras.Input(shape=(sentence_sequence_length,bert_embedding_length),batch_size=batch_size, dtype="float32", name="BERT Input Layer")
    x = Concatenate()([bert_input,BiLSTM_outputs])

    output = sec_conv_BiLSTM(
        window_length = 10,
        sentence_encoding_length = bert_embedding_length+256)(x)


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
    keras.utils.plot_model(model,to_file="model_out.png",show_shapes=True,expand_nested=True)



def instantiate_bert_client():
    pass

# Assembles label prediction network
def build_label_prediction_network():
    pass

