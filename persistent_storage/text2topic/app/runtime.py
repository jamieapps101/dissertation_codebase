import tensorflow as tf
import model_generation
import os
from tensorflow import keras

# This file loads up the trained topic segmentation model
# data from speech to text unit should be sent directly here,
# and this program will send off to word2vec and bert to encode.
# this is to preserve data order.

MODEL_WEIGHTS_PATH = None


def load_topic_segment_model(weights_path,epoch):
    # load in model structure
    raw_inputs,outputs = model_generation.build_Att_BiLSTM(
        batch_size=1,
        masking_enabled=True)
    model = keras.Model(raw_inputs, outputs)

    # load in model weights
    model_path = os.path.join(weights_path,"model_epoch_{}".format(epoch))
    if os.path.exists(weights_path):
        print("loading model weights from:\n{}".format(model_path))
        model.load_weights(model_path)
    else:
        print("path:\n{}\ndoes not exist".format(weights_path))
        return None
    return model


if __name__=="__main__":
    # get model
    model = load_topic_segment_model(MODEL_WEIGHTS_PATH,10)
    if model is None:
        exit(1)

    # start up MQTT connection
    