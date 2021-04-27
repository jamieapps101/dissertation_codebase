# This script puts togeather a model based on "Improving Context Modeling in Neural Topic Segmentation"

# Obvs:
import tensorflow as tf

# Required for BERT
import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

import keras from tf
from keras.layers import LSTM,Embedding,Dense,Bidirectional

# top level function, to be used by external scripts
# This function calls all remaining functions in this file and
# returns a fully build model
def build_model():
    pass

# combines Att BiLSTM and bert models
def build_sentence_encoding_network():
    pass

def build_Att_BiLSTM(bert_total_word_encodings):
    model = keras.Sequential()
    model.add

def instantiate_bert_client():
    pass

# Assembles label prediction network
def build_label_prediction_network():
    pass

