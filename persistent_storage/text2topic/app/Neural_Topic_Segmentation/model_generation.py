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
# import tensorflow_datasets as tfds
# tfds.disable_progress_bar()


from tensorflow import keras
from tensorflow.keras.layers import LSTM,Embedding,Dense,Multiply,Bidirectional,Softmax,Dot,Attention,MaxPooling1D,Reshape,Add,Concatenate,ConvLSTM2D,Subtract,RepeatVector,TimeDistributed
from tensorflow.keras import activations
from tensorflow import linalg
from tensorflow.keras.utils import plot_model


# Sentence encoding BiLSTM
class se_BiLSTM(keras.layers.Layer):
    def __init__(self, return_sequences=False):
        super(se_BiLSTM, self).__init__()
        self.x_forward = LSTM(
            units = 256,
            recurrent_activation=activations.tanh, # check this
            return_sequences=return_sequences,
            go_backwards=False,
            stateful=True,
        )
        self.x_backward = LSTM(
            units = 256,
            recurrent_activation=activations.tanh, # check this
            return_sequences=return_sequences,
            go_backwards=True,
            stateful=True,
        )
        self.add=Add()

    def call(self, input_t):
        return self.add([self.x_forward(input_t),self.x_backward(input_t)])


class custom_SAT(keras.layers.Layer):
    def __init__(self, output_vec_len = 300):
        super(custom_SAT, self).__init__()
        self.dense0 = Dense(units=output_vec_len, activation=activations.tanh)

        initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        self.u_w = tf.Variable(initial_value=initializer((output_vec_len,output_vec_len,)),
                               trainable=True)
        self.softmax   = Softmax()

    def call(self, x):
        x = self.dense0(x)
        x = self.softmax(linalg.matmul(x,self.u_w))
        x = tf.math.reduce_sum(x, axis=-2, keepdims=False, name=None)
        return x

class se_Att_BiLSTM(keras.layers.Layer):
    def __init__(self, output_vec_len = 300):
        super(se_Att_BiLSTM, self).__init__()
        self.bistm_0 = se_BiLSTM(return_sequences=True)
        self.bistm_1 = se_BiLSTM(return_sequences=True)
        self.sat     = custom_SAT()

    def call(self, x):
        x = self.bistm_0(x)
        x = self.bistm_1(x)
        x = self.sat(x)
        return x

# Sentence encoding Comparison BiLSTM
class se_comp_BiLSTM(keras.layers.Layer):
    def __init__(self, lstm_units = 256):
        super(se_comp_BiLSTM, self).__init__()
        self.x_forward = LSTM(
            units = lstm_units,
            recurrent_activation=activations.tanh, # check this
            return_sequences=True,
            go_backwards=False,
            stateful=True,
        )
        self.x_backward = LSTM(
            units = lstm_units,
            recurrent_activation=activations.tanh, # check this
            return_sequences=True,
            go_backwards=True,
            stateful=True,
        )
        self.add=Add()

    def call(self, input_t):
        return self.add([self.x_forward(input_t),self.x_backward(input_t)])

# This item collects previous inputs, and performes the selective self attention
# while returning outputs of the same shape for lstm after wards
class RSA_layer(keras.layers.Layer):
    def __init__(self, window_size,input_units):
        self_attention_output_units = input_units
        self.input_units = input_units
        super(RSA_layer, self).__init__()
        self.window_size = window_size
        self.state = tf.zeros([input_units,window_size],dtype=tf.float32)
        temp = np.zeros((window_size,window_size-1))
        for i in range(window_size-1):
            temp[i+1,i]=1
        self.state_shift = tf.constant(temp,dtype=tf.float32)

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_units*2+1, self_attention_output_units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self_attention_output_units,), dtype="float32"), 
            trainable=True,
        )

       

    def call(self, input_tensor):
        # shift state back one
        # self.state 256,5
        # self.state_shift 5,4
        shifted_state = tf.matmul(self.state,self.state_shift) # 256,4
        transposed_input_tensor = tf.transpose(input_tensor) # 256,1
        self.state = tf.concat([shifted_state,transposed_input_tensor],axis=-1) # 256,5
        
        # helps make the maths prettier
        flip_state = tf.transpose(self.state) # 5,256

        h_i = tf.stack([flip_state for i in range(self.window_size)],axis=0) # 5n,5,256
        h_j = tf.stack([flip_state for i in range(self.window_size)],axis=1) # 5,5n,256
        h_i_dot_h_j = tf.math.reduce_sum(tf.multiply(h_i,h_j),axis=2,keepdims=True) # 5,5,1
        

        se_ij = tf.concat([h_i,h_j,h_i_dot_h_j],axis=-1) # 5,5,513
        # add in dense layer for each h_ij. sam layer applied
        # to every i,j combination
        sim_ij = tf.matmul(se_ij,self.w)+self.b # 5,5,256
        a_ij = tf.nn.softmax(sim_ij,axis=1) # 5,5(norm on this axis),256

        # TODO, try replacing this with h_j
        c_i = tf.math.reduce_sum(tf.math.multiply(h_i,a_ij),axis=1) # 5,256
        c_last = c_i[-1,:]
        c_last_reshaped = tf.reshape(c_last,[1,self.input_units]) 
        # return tf.reshape(c_last,[self.input_units,1])
        return c_last_reshaped


# class AuxTaskModule(keras.layers.Layer):
#     def __init__(self, window_size,input_units):
#         super(AuxTaskModule, self).__init__()
#         self.state = tf.zeros([input_units,window_size],dtype=tf.float32)

#     def call(self, input_tensor):
#         # shift state back one
#         # self.state 256,5
#         # self.state_shift 5,4
#         shifted_state = tf.matmul(self.state,self.state_shift) # 256,4
#         transposed_input_tensor = tf.transpose(input_tensor) # 256,1
#         self.state = tf.concat([shifted_state,transposed_input_tensor],axis=-1) # 256,5
        
#         # helps make the maths prettier
#         flip_state = tf.transpose(self.state) # 5,256

#         h_i = tf.stack([flip_state for i in range(self.window_size)],axis=0) # 5n,5,256
#         h_j = tf.stack([flip_state for i in range(self.window_size)],axis=1) # 5,5n,256
#         h_i_dot_h_j = tf.math.reduce_sum(tf.multiply(h_i,h_j),axis=2,keepdims=True) # 5,5,1
        

#         se_ij = tf.concat([h_i,h_j,h_i_dot_h_j],axis=-1) # 5,5,513
#         # add in dense layer for each h_ij. sam layer applied
#         # to every i,j combination
#         sim_ij = tf.matmul(se_ij,self.w)+self.b # 5,5,256
#         a_ij = tf.nn.softmax(sim_ij,axis=1) # 5,5(norm on this axis),256

#         # TODO, try replacing this with h_j
#         c_i = tf.math.reduce_sum(tf.math.multiply(h_i,a_ij),axis=1) # 5,256
#         c_last = c_i[-1,:]
#         c_last_reshaped = tf.reshape(c_last,[1,self.input_units]) 
#         # return tf.reshape(c_last,[self.input_units,1])
#         return c_last_reshaped


# build sentence level BiLSTM layers
def build_Att_BiLSTM(
    embedding_length = 300,
    bert_embedding_length=1024, 
    batch_size=1,
    self_attention_enabled=True):

    # input to this should be [None,None,None,300]
    # 1 - None - batch size
    # 2 - None - no of sentence
    # 3 - None - no of words
    # 4 - 300 - word encoding size
    word2vec_input = keras.Input(shape=(None,None,embedding_length), batch_size=batch_size,dtype="float32",name="Embedding Input Layer")
    bert_input = keras.Input(shape=(None,bert_embedding_length),batch_size=batch_size, dtype="float32", name="BERT Input Layer")
    se_out = TimeDistributed(se_Att_BiLSTM())(word2vec_input)
    se_bert_out = Concatenate()([bert_input,se_out])

    lstm_units = 256
    se_comp_bilstm1_out = se_comp_BiLSTM(lstm_units)(se_bert_out)

    # create a buffer to store previous sentence encodings from this section
    # add time distributed, as it removes the None element in the lstm output, 
    # allowing the tensorss to be processed individually
    t_stacker = RSA_layer(window_size=5,input_units=lstm_units)
    sentence_encodings_stack = TimeDistributed(t_stacker)(se_comp_bilstm1_out)
    

    # sentence_encodings_stack = TimeDistributed(d)(se_comp_bilstm1_out)
    out = tf.concat([se_comp_bilstm1_out,sentence_encodings_stack],axis=-1)
    se_comp_bilstm2_out = se_comp_BiLSTM(lstm_units)(out)

    return [word2vec_input,bert_input],se_comp_bilstm2_out


def build_Att_BiLSTM2(
    embedding_length = 300,
    bert_embedding_length=1024, 
    batch_size=1,
    self_attention_enabled=True):
    lstm_units=embedding_length
    word2vec_input = keras.Input(shape=(embedding_length), batch_size=batch_size,dtype="float32",name="Embedding Input Layer")
    t_stacker = RSA_layer(window_size=5,input_units=lstm_units)
    out = t_stacker(word2vec_input)
    return word2vec_input,out


if __name__=="__main__":
    print("Running from {}".format(__file__))
    

    raw_inputs,outputs = model_gen.build_Att_BiLSTM()

    model = keras.Model(raw_inputs, outputs)
    model.summary()
    # model.summary(line_length=160)
    keras.utils.plot_model(model,to_file="model_out.png",show_shapes=True,expand_nested=True)





def instantiate_bert_client():
    pass

# Assembles label prediction network
def build_label_prediction_network():
    pass

