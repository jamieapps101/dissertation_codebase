# import libs
import os
os.environ['MPLCONFIGDIR'] = '/app/matplotlib_temp'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.executing_eagerly()
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import LSTM,Embedding,Dense,Multiply,Bidirectional,Softmax,Dot,Attention,MaxPooling1D,Reshape,Add,Concatenate,ConvLSTM2D,Subtract,RepeatVector,TimeDistributed
from tensorflow.keras import activations
from tensorflow import linalg
from tensorflow.keras.utils import plot_model
from datetime import datetime,timezone
import time


# import from other files
import model_generation


# pulled in from other file
max_sentence_len = 90
max_sentences_per_sample = 100 # this is for both segments
word2vec_encoding_len = 300
bert_encoding_len = 1024
max_buffer_length = 250
# data_we_shape = [actual_buffer_length,max_sentences_per_sample,max_sentence_len,word2vec_encoding_len]
# data_se_shape = [actual_buffer_length,max_sentences_per_sample,bert_encoding_len]
# data_gt_shape = [actual_buffer_length,max_sentences_per_sample,1]
data_we_shape = [max_sentences_per_sample,max_sentence_len,word2vec_encoding_len]
data_se_shape = [max_sentences_per_sample,bert_encoding_len]
data_gt_shape = [max_sentences_per_sample,1]

# data processing left incorrect tensor ordering, so shapes have to be juggled
correction_needed = False
if correction_needed:
    data_se_shape = [max_sentences_per_sample,max_sentence_len,word2vec_encoding_len]
    data_we_shape = [max_sentences_per_sample,bert_encoding_len]

USE_GPU = True
if USE_GPU:
    # stop mem errors
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    tf.config.experimental.set_visible_devices([], 'GPU')

# turn off meta optimiser
tf.config.optimizer.set_experimental_options(
    {"disable_meta_optimizer":True}
)


def read_tfrecord(serialized_example):
  feature_description = {
        'se': tf.io.FixedLenFeature([], tf.string),
        'we': tf.io.FixedLenFeature([], tf.string),
        'gt': tf.io.FixedLenFeature([], tf.string)
  }
  example = tf.io.parse_single_example(serialized_example, feature_description)
  se = tf.io.parse_tensor(example['se'], out_type = tf.float64)
  se = tf.reshape(se,data_se_shape)
  
  we = tf.io.parse_tensor(example['we'], out_type = tf.float64)
  we = tf.reshape(we,data_we_shape)
  
  gt = tf.io.parse_tensor(example['gt'], out_type = tf.float64)
  gt = tf.reshape(gt,data_gt_shape)
  return (we,se,gt)


def get_values(we,se,gt):
    tensor = {}
    tensor["WE"] = we
    tensor["SE"] = se
    return tensor

def get_labels(we,se,gt):
    tensor = {}
    tensor["boundry_out"] = gt
    return tensor

def get_split(we,se,gt):
    tensor = {}
    tensor["WE"] = we
    tensor["SE"] = se
    tensor["boundry_out"] = gt
    return tensor

dataset_dir = "/app/data/processed/data_8/"
batch_size  = 8

content = ["city"]
# content = ["disease","city"]
types   = ["train","test","validation"]
# items = [7,2,1]
items = [55,16,8]


def fetch_data():
    print("arranging input pipeline")
    datasets = {}
    for c in content:
        datasets[c] = {}
        for shards,t in zip(items,types):
            datasets[c][t] = {}
            files = ["{}_{}_{}.tfrecord".format(c,t,i) for i in range(shards)]
            raw_dataset      = tf.data.TFRecordDataset([dataset_dir+file_name for file_name in files])
            mapped_dataset   = raw_dataset.map(read_tfrecord)
            batched_dataset  = mapped_dataset.batch(batch_size,drop_remainder=True)
            buffered_dataset = batched_dataset.prefetch(buffer_size=1) # load in 1 batch ahead of time
            datasets[c][t]   = buffered_dataset
    return datasets

# define tf functions
@tf.function
def train_step(we,se, gt):
    with tf.GradientTape() as tape:
        logits = model({"WE":we,"SE":se}, training=True)
        loss_value = loss_fn(gt, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(gt, logits)
    return loss_value

@tf.function
def test_step(we,se, gt):
    val_logits = model({"WE":we,"SE":se}, training=False)
    val_acc_metric.update_state(gt, val_logits)


# Train model with manual trianing loops
# Instantiate an optimizer.
lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=1e-2,
    decay_steps=100,
    decay_rate=0.8, # sweeps from 0.01 to 0.0001 over the 10 epochs
    staircase=False)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
# optimizer = keras.optimizers.Adam(learning_rate=0.1)
# Instantiate a loss function.
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

current_dir = "{:%m_%d_%H_%M}".format(datetime.now(timezone.utc))
model_save_path = "/app/data/models/"+current_dir
print("checkpoints stored in:\n\t{}".format(model_save_path))
os.makedirs(model_save_path,exist_ok=True)
# Prepare the metrics.
train_acc_metric    = keras.metrics.BinaryCrossentropy()
train_loss_metric   = keras.metrics.BinaryCrossentropy()
val_acc_metric      = keras.metrics.BinaryCrossentropy()
val_los_metric      = keras.metrics.BinaryCrossentropy()

history = pd.DataFrame({
    'validation_acc': np.array([]),
    'train_loss': np.array([]),
    'train_acc': np.array([]),
    'epoch': np.array([]),
})

# setup training params
# Instantiate an optimizer.
# get model
model=None
# pass in dict of path to dir and epoch number:
# {
#     path: "",
#     epoch: 5,
# }
def get_model(load_weights_from=None):
    raw_inputs,outputs = model_generation.build_Att_BiLSTM(
        batch_size=batch_size,
        masking_enabled=True)
    model = keras.Model(raw_inputs, outputs)

    # run training loop
    print("Training this:")
    model.summary()
    keras.utils.plot_model(model,to_file="model_out.png",show_shapes=True,expand_nested=True)

    if load_weights_from is not None:
        model_path = os.path.join(load_weights_from["path"],"model_epoch_{}".format(load_weights_from["epoch"]))
        if os.path.exists(load_weights_from["path"]):
            print("loading model weights from:\n{}".format(model_path))
            model.load_weights(model_path)
        else:
            print("path:\n{}\ndoes not exist".format(load_weights_from["path"]))
    # model.compile(run_eagerly=True)
    return model



def train_model(epochs):
    global history
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (we_batch,se_batch,gt_batch) in enumerate(datasets["city"]["train"]):
            loss_value = train_step(we_batch,se_batch,gt_batch)

            # Log every 10 batches.
            if step % 20 == 0:
                print("Step {}, per batch training loss: {:.6}, total training samples: {}".format(step, float(loss_value),(step + 1) * batch_size ))
            # if step == 20:
                # break
        model.save_weights(os.path.join(model_save_path,"model_epoch_{}".format(epoch)))
        # model.save(os.path.join(model_save_path,"model_whole_epoch_{}".format(epoch)))
        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: {:.4}".format(float(train_acc),))
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
        # Run a validation loop at the end of each epoch.
        for (we_val_batch,se_val_batch,gt_val_batch) in datasets["city"]["validation"]:
            test_step(we_val_batch,se_val_batch,gt_val_batch)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
        history = history.append({
            'validation_acc': val_acc,
            'train_loss': loss_value,
            'train_acc': train_acc,
            'epoch': epoch,
        }, ignore_index=True)

if __name__=="__main__":
    


    # load data
    datasets = fetch_data()

    # get model
    model = get_model({"path":"/app/data/models/04_04_12_26","epoch":0})

    epochs = 10
    train_model(epochs)


    # save model
    model.save_weights(os.path.join(model_save_path,"model_final"))
    # model.save_weights(os.path.join(model_save_path,"model_whole_final"))
    # save history
    history.to_csv(os.path.join(model_save_path,"model_hist.csv"))