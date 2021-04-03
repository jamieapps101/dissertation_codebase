# This script puts togeather a model based on "Improving Context Modeling in Neural Topic Segmentation"

import os
os.environ['MPLCONFIGDIR'] = '/app/matplotlib_temp'
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

from tensorflow import keras
from tensorflow.keras.layers import LSTM,Embedding,Dense,Multiply,Bidirectional,Softmax,Dot,Attention,MaxPooling1D,Reshape,Add,Concatenate,ConvLSTM2D,Subtract,RepeatVector,TimeDistributed,Masking, Reshape
from tensorflow.keras import activations
from tensorflow import linalg
from tensorflow.keras.utils import plot_model


# Sentence encoding BiLSTM
class se_BiLSTM(keras.layers.Layer):
    def __init__(self, return_sequences=False):
        super(se_BiLSTM, self).__init__()
        self.x_forward = LSTM(
            units = 256,
            activation='tanh',
            recurrent_activation='sigmoid', # check this
            return_sequences=return_sequences,
            go_backwards=False,
            stateful=True,
        )
        self.x_backward = LSTM(
            units = 256,
            activation='tanh',
            recurrent_activation='sigmoid', # check this
            return_sequences=return_sequences,
            go_backwards=True,
            stateful=True,
        )
        self.add=Add()

    def call(self, input_t,mask=None):
        return self.add([self.x_forward(input_t,mask=mask),self.x_backward(input_t,mask=mask)])


class custom_SAT(keras.layers.Layer):
    def __init__(self, output_vec_len = 300):
        super(custom_SAT, self).__init__()
        self.dense0 = Dense(units=output_vec_len, activation=activations.tanh)

        initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        self.u_w = tf.Variable(initial_value=initializer((output_vec_len,output_vec_len,)),
                               trainable=True)

    def call(self, x):
        x = self.dense0(x)
        x = tf.math.softmax(linalg.matmul(x,self.u_w), axis=-2)
        x = tf.math.reduce_sum(x, axis=-2, keepdims=False, name=None)
        return x

class se_Att_BiLSTM(keras.layers.Layer):
    def __init__(self, output_vec_len = 300,**kwargs):
        super(se_Att_BiLSTM, self).__init__(**kwargs)
        self.masking = Masking(mask_value=0.0)
        self.dense_0 = Dense(units=300)
        self.lstm_0 = LSTM(
            units = output_vec_len,
            activation='tanh',
            recurrent_activation='sigmoid', # check this
            return_sequences=True,
            stateful=True,
            name="lstm_0",
        )
        self.bilstm_0 = tf.keras.layers.Bidirectional(self.lstm_0, merge_mode='sum')

        self.lstm_1 = LSTM(
            units = output_vec_len,
            activation='tanh',
            recurrent_activation='sigmoid', # check this
            return_sequences=True,
            stateful=True,
            name="lstm_1",
        )
        self.bilstm_1 = tf.keras.layers.Bidirectional(self.lstm_1, merge_mode='sum')

        
        
        # self.bistm_1 = se_BiLSTM(return_sequences=True)
        self.sat     = custom_SAT()

    def call(self, x):
        z = self.masking(x)
        y = self.dense_0(z)
        # mask = self.masking.compute_mask(x)
        a = self.bilstm_0(y)
        b = self.bilstm_0(a)
        # a = self.add([
        #     self.x_forward_0(z,mask=mask),
        #     self.x_backward_0(z,mask=mask)
        #     ])
        # b = self.bistm_1(a)
        # b = self.add([self.x_forward_1(a),self.x_backward_1(a)])
        c = self.sat(b)
        return c

# Sentence encoding Comparison BiLSTM
class se_comp_BiLSTM(keras.layers.Layer):
    def __init__(self, lstm_units = 256):
        super(se_comp_BiLSTM, self).__init__()
        self.masking = Masking(mask_value=0.0)
        self.x_forward = LSTM(
            units = lstm_units,
            # recurrent_activation=activations.tanh, # check this
            return_sequences=True,
            go_backwards=False,
            stateful=True,
        )
        self.x_backward = LSTM(
            units = lstm_units,
            # recurrent_activation=activations.tanh, # check this
            return_sequences=True,
            go_backwards=True,
            stateful=True,
        )
        self.add=Add()

    def call(self, input_t):
        x = self.masking(input_t)
        return self.add([self.x_forward(x),self.x_backward(x)])

# This item collects previous inputs, and performes the selective self attention
# while returning outputs of the same shape for lstm after wards
class RSA_layer(keras.layers.Layer):
    def __init__(self, window_size,input_units,batch_size=64):
        self_attention_output_units = input_units
        self.input_units = input_units
        super(RSA_layer, self).__init__()
        self.window_size = window_size
        self.batch_size = batch_size

        state_init = tf.zeros_initializer()
        self.state = tf.Variable(
            initial_value=state_init(shape=(batch_size,input_units,window_size), dtype="float32"), 
            trainable=False,
        )

        # self.state = tf.zeros([batch_size,input_units,window_size],dtype=tf.float32)
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
        shifted_state = tf.matmul(self.state,self.state_shift) # <batch_size>,None,256,4
        transposed_input_tensor = tf.reshape(input_tensor,[self.batch_size,self.input_units,1]) # <batch_size>,1,256,1
        # cloned_transposed_input_tensor = tf.identity(transposed_input_tensor)
        new_state = tf.concat([shifted_state,transposed_input_tensor],axis=-1,name="concat_layer_aaah") # <batch_size>,1,256,5
        

        # helps make the maths prettier
        flip_state = tf.reshape(new_state,[self.batch_size,self.window_size,self.input_units]) # <batch_size>,1,5,256

        h_i = tf.stack([flip_state for i in range(self.window_size)],axis=1)   # <batch_size>,5n,5,256
        h_j = tf.stack([flip_state for i in range(self.window_size)],axis=2) # <batch_size>,5,5n,256
        h_i_dot_h_j = tf.math.reduce_sum(tf.multiply(h_i,h_j),axis=3,keepdims=True) # <batch_size>,1,5,5,1
        

        se_ij = tf.concat([h_i,h_j,h_i_dot_h_j],axis=-1) # <batch_size>,1,5,5,513
        # add in dense layer for each h_ij. sam layer applied
        # to every i,j combination
        sim_ij = tf.matmul(se_ij,self.w)+self.b # <batch_size>,1,5,5,256
        a_ij = tf.nn.softmax(sim_ij,axis=-2) # <batch_size>,1,5,5(norm on this axis),256


        # h_i - 1700: 0.692493
        # TODO, try replacing this with h_j
        c_i = tf.math.reduce_sum(tf.math.multiply(h_i,a_ij),axis=-2) # <batch_size>,1,5,256
        c_last = c_i[:,-1,:]
        c_last_reshaped = tf.reshape(c_last,[self.batch_size,self.input_units]) 
        
        tf.keras.backend.update(self.state, new_state)


        return c_last_reshaped

# Sentence encoding Comparison BiLSTM
class custom_dense(keras.layers.Layer):
    def __init__(self, output_neurons=256,activation=tf.keras.activations.relu, **kwargs):
        super(custom_dense, self).__init__( **kwargs)
        self.output_neurons = output_neurons
        self.activation = activation
        
    def build(self, input_shape):
        if int(input_shape[-1])==1:
            input_dim = input_shape[-2]
        else:
            input_dim = input_shape[-1]
        
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, self.output_neurons), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.output_neurons), dtype="float32"), 
            trainable=True,
        )

    def call(self, input_t):
        shape = list(input_t.shape)
        if int(shape[-1])==1: # if there is an extra 1 dim
            return self.activation(tf.matmul(tf.reshape(input_t,shape[:-1]),self.w)+self.b)
        else: # otherwise if everything is normal
            return self.activation(tf.matmul(input_t,self.w)+self.b)

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
    word2vec_input = keras.Input(shape=(None,None,embedding_length), batch_size=batch_size,dtype="float32",name="WE")
    # word2vec_input = Masking(mask_value=0.0)(word2vec_input)
    bert_input = keras.Input(shape=(None,bert_embedding_length),batch_size=batch_size, dtype="float32", name="SE")
    # bert_input = Masking(mask_value=0.0)(bert_input)
    se_out = TimeDistributed(se_Att_BiLSTM(name="self_att_bilstm_hello"))(word2vec_input)
    se_bert_out = Concatenate()([bert_input,se_out])

    lstm_units = 256
    se_comp_bilstm1_out = se_comp_BiLSTM(lstm_units)(se_bert_out)

    # create a buffer to store previous sentence encodings from this section
    # add time distributed, as it removes the None element in the lstm output, 
    # allowing the tensorss to be processed individually
    t_stacker = RSA_layer(window_size=5,input_units=lstm_units,batch_size=batch_size)
    sentence_encodings_stack = TimeDistributed(t_stacker)(se_comp_bilstm1_out)
    
    out = tf.concat([se_comp_bilstm1_out,sentence_encodings_stack],axis=-1)
    se_comp_bilstm2_out = se_comp_BiLSTM(lstm_units)(out)
    # reshaper = Reshape([256,1])
    # se_comp_bilstm2_out_reshaped = TimeDistributed(reshaper)(se_comp_bilstm2_out)


    dense_0_out = custom_dense(output_neurons=lstm_units,name="dense_0")(se_comp_bilstm2_out)
    # reshape_0_o/ut = Reshape([None, 256,1])(dense_0_out)
    dense_1_out = custom_dense(output_neurons=1,name="dense_1")(dense_0_out)

    softmax_out = Softmax(name="boundry_out",axis=-2)(dense_1_out)

    return [word2vec_input,bert_input],softmax_out


if __name__=="__main__":
    print("Running from {}".format(__file__))
    
    batch_size=64


    # lstm_units = 256
    # word2vec_input = keras.Input(shape=(None,lstm_units), batch_size=batch_size,dtype="float32",name="WE")
    # # t_stacker = RSA_layer(window_size=5,input_units=lstm_units)
    # # sentence_encodings_stack = TimeDistributed(t_stacker)(word2vec_input)
    # seab = se_Att_BiLSTM() 
    # out = seab(word2vec_input)
    # model = keras.Model(word2vec_input, out)
    

    raw_inputs,outputs = build_Att_BiLSTM(batch_size=batch_size)
    model = keras.Model(raw_inputs, outputs)
    model.summary()
    # model.summary(line_length=160)
    # keras.utils.plot_model(model,to_file="model_out.png",show_shapes=True,expand_nested=True)





def instantiate_bert_client():
    pass

# Assembles label prediction network
def build_label_prediction_network():
    pass

