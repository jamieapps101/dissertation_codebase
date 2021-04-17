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


# [1] - https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765

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

dataset_dir = "/app/data/processed/data_9/"
batch_size  = 8
# content = ["disease"]
content = ["city"]
# content = ["disease","city"]
types   = ["train","test","validation"]
# items = [11,3,2]
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


# setup training params
# Instantiate an optimizer.
# get model

# pass in dict of path to dir and epoch number:
# {
#     path: "",
#     epoch: 5,
# }
def get_model(load_weights_from=None,masking_enabled=True):
    raw_inputs,outputs = model_generation.build_Att_BiLSTM(
        batch_size=batch_size,
        masking_enabled=masking_enabled)
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
    

if __name__=="__main__":
    # load data
    datasets = fetch_data()

    # Train model with manual training loops
    # Instantiate an optimizer.
    lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.5,
        staircase=True)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer = keras.optimizers.Adam(learning_rate=0.001)
    # optimizer = keras.optimizers.SGD(learning_rate=0.01)
    # Instantiate a loss function.
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    # loss_fn = model_generation.CustomLossFunction()

    current_dir = "{:%m_%d_%H_%M}".format(datetime.now(timezone.utc))
    model_save_path = "/app/data/models/"+current_dir

    ## setup tensorboard stuff
    model_logging_path       = os.path.join("/app/data/logs",current_dir)
    model_logging_path_train = os.path.join(model_logging_path,"train")
    model_logging_path_test  = os.path.join(model_logging_path,"test")
    model_logging_path_lr    = os.path.join(model_logging_path,"lr")
    paths = [
        model_save_path,
        model_logging_path,
        model_logging_path_train,
        model_logging_path_test,
    ]
    train_summary_writer = tf.summary.create_file_writer(model_logging_path_train,name="train")
    test_summary_writer  = tf.summary.create_file_writer(model_logging_path_test,name="test")
    lr_summary_writer    = tf.summary.create_file_writer(model_logging_path_lr,name="Learn-Rate")
    print("checkpoints stored in:\n\t{}".format(model_save_path))
    for path in paths:
        os.makedirs(path,exist_ok=True)
    # Define our metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_accuracy  = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
    train_loss     = tf.keras.metrics.BinaryCrossentropy('train_loss', dtype=tf.float32,from_logits=True)
    test_loss      = tf.keras.metrics.BinaryCrossentropy('test_loss',  dtype=tf.float32,from_logits=True)
    learn_rate     = tf.keras.metrics.Mean('Learn-Rate', dtype=tf.float32)
    
    # enable beast mode debugging
    # tf.debugging.experimental.enable_dump_debug_info(
    #     dump_root=model_logging_debugger_path,
    #     tensor_debug_mode="FULL_HEALTH",
    #     circular_buffer_size=1000) # save only the last 1000 tensors

    history = pd.DataFrame({
        'validation_acc': np.array([]),
        'train_loss': np.array([]),
        'train_acc': np.array([]),
        'epoch': np.array([]),
    })


    # define tf functions
    @tf.function
    def train_step(we,se, gt):
        with tf.GradientTape() as tape:
            logits = model({"WE":we,"SE":se}, training=True)
            loss_value = loss_fn(gt, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_accuracy.update_state(gt, logits)
        train_loss.update_state(gt, logits)
        return loss_value

    @tf.function
    def test_step(we,se, gt):
        val_logits = model({"WE":we,"SE":se}, training=False)
        test_accuracy.update_state(gt, val_logits)
        test_loss.update_state(gt, val_logits)


    model = get_model(masking_enabled=True)

    epochs = 10

    print("initial learn rate: {}".format(optimizer.learning_rate(0)))

    total_steps = 0

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # get a snapshot of weights before the training epoch
        before = model.trainable_variables

        # Iterate over the batches of the dataset. from [1]
        step_start_time = time.time()
        ave_step_time = 0
        for step, (we_batch,se_batch,gt_batch) in enumerate(datasets["city"]["train"]):
            loss_value = train_step(we_batch,se_batch,gt_batch)
            step_time = time.time()-step_start_time
            ave_step_time = (3*ave_step_time+step_time)/4
            # Log every 20 batches.
            if step % 20 == 0:
                print("Step {}, \n per batch training loss: {:.8}\ntotal training samples: {},\nave step time: {:.1}s".format(step, float(loss_value),(step + 1) * batch_size, ave_step_time))
                # get a snapshot of after the epoch, and compare from[1]
                layers_count = 0
                changed_layers_count = 0
                after = model.trainable_variables
                for b, a in zip(before, after):
                    layers_count+=1
                    # Make sure something changed.
                    if np.any(b.numpy() != a.numpy()):
                        print("{} has not changed".format(a.name))
                        changed_layers_count+=1
                print("{}/{} layers with no changed weights".format(changed_layers_count,layers_count))
                print("current lr: {:.5}".format(optimizer.learning_rate(total_steps)))
                print("")
                before = None
                before = after

                # log every 20 steps
                # tensorboard logging stuff
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss',     train_loss.result(), step=total_steps)
                    tf.summary.scalar('accuracy', train_accuracy.result(), step=total_steps)

                with lr_summary_writer.as_default():
                    tf.summary.scalar('Learn-Rate',  optimizer.learning_rate(total_steps), step=total_steps)

                ## Reset metrics every epoch
                train_loss.reset_states()
                train_accuracy.reset_states()
            
            # run validation every 1000 steps
            if step % 1000 == 0:
                for step,(we_val_batch,se_val_batch,gt_val_batch) in enumerate(datasets["city"]["validation"]):
                    test_step(we_val_batch,se_val_batch,gt_val_batch)
                # val_acc = test_accuracy.result()
                # print("Validation acc: {:.4}" .format(float(val_acc)))
                # print("Time taken:     {:.2}s".format(time.time() - start_time))
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss',     test_loss.result(), step=total_steps)
                    tf.summary.scalar('accuracy', test_accuracy.result(), step=total_steps)
                test_loss.reset_states()
                test_accuracy.reset_states()
            total_steps+=1
            

            
            # quit after 100 batches
            # if step >= 100:
            #     break
            
        model.save_weights(os.path.join(model_save_path,"model_epoch_{}".format(epoch)))
        # model.save(os.path.join(model_save_path,"model_whole_epoch_{}".format(epoch)))
        # Display metrics at the end of each epoch.
        # train_acc = train_accuracy.result()
        # print("Training acc over epoch: {:.4}".format(float(train_acc),))
        # Reset training metrics at the end of each epoch
        # Run a validation loop at the end of each epoch.
        for step,(we_val_batch,se_val_batch,gt_val_batch) in enumerate(datasets["city"]["validation"]):
            test_step(we_val_batch,se_val_batch,gt_val_batch)
        # val_acc = test_accuracy.result()
        # print("Validation acc: {:.4}" .format(float(val_acc)))
        # print("Time taken:     {:.2}s".format(time.time() - start_time))
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=total_steps)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=total_steps)
        test_loss.reset_states()
        test_accuracy.reset_states()
        

        






    # save model
    model.save_weights(os.path.join(model_save_path,"model_final"))
    # model.save_weights(os.path.join(model_save_path,"model_whole_final"))
    # save history
    history.to_csv(os.path.join(model_save_path,"model_hist.csv"))