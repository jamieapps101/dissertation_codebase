# import libs
import os
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
from tensorflow import keras
from tensorflow.keras.layers import LSTM,Embedding,Dense,Multiply,Bidirectional,Softmax,Dot,Attention,MaxPooling1D,Reshape,Add,Concatenate,ConvLSTM2D,Subtract,RepeatVector,TimeDistributed
from tensorflow.keras import activations
from tensorflow import linalg
from tensorflow.keras.utils import plot_model
from datetime import datetime,timezone
import time


# import from other files
import model_generation


def get_and_zip_data(files,dataset_dir):
    train_we_ds      = tf.data.TFRecordDataset([dataset_dir+files[0] for file_name in train_we_files]).map(lambda x: tf.io.parse_tensor(x, tf.float64))
    train_se_ds      = tf.data.TFRecordDataset([dataset_dir+files[1] for file_name in train_se_files]).map(lambda x: tf.io.parse_tensor(x, tf.float64))
    train_gt_ds      = tf.data.TFRecordDataset([dataset_dir+files[2] for file_name in train_gt_files]).map(lambda x: tf.io.parse_tensor(x, tf.float64))
    return tf.data.Dataset.zip((train_we_ds,train_se_ds,train_gt_ds))

if __name__=="__main__":
    # stop mem errors
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # load data
    dataset_dir = "/app/data/processed/prev/"
    files = os.listdir(dataset_dir)
    train_files = [file_name for file_name in files if "train" in file_name]
    test_files = [file_name for file_name in files if "test" in file_name]
    validation_files = [file_name for file_name in files if "validation" in file_name]

    # overide this for testing
    # train_data = []
    # for i in range(7):
    #     train_we_files      = ["disease_train_we_{}.tfrecord".format(i)]
    #     train_se_files      = ["disease_train_se_{}.tfrecord".format(i)]
    #     train_gt_files      = ["disease_train_gt_{}.tfrecord".format(i)]
    #     file_names = train_we_files+train_se_files+train_gt_files
    #     zipped_data = get_and_zip_data(files,dataset_dir)
    #     train_data.append(zipped_data)


    content = ["disease","city"]
    types   = ["train","test","validation"]
    items = [7,2,1]
    datasets = {}
    for c in content:
        datasets[c] = {}
        for shards,t in zip(items,types):
            files_we = []
            files_se = []
            files_gt = []
            for i in range(shards):
                files_we.append("{}_{}_we_{}.tfrecord".format(c,t,i))
                files_se.append("{}_{}_se_{}.tfrecord".format(c,t,i))
                files_gt.append("{}_{}_gt_{}.tfrecord".format(c,t,i))
            dataset_we = tf.data.TFRecordDataset([dataset_dir+file_name for file_name in files_we]).map(lambda x: tf.io.parse_tensor(x, tf.float64))
            dataset_se = tf.data.TFRecordDataset([dataset_dir+file_name for file_name in files_se]).map(lambda x: tf.io.parse_tensor(x, tf.float64))
            dataset_gt = tf.data.TFRecordDataset([dataset_dir+file_name for file_name in files_gt]).map(lambda x: tf.io.parse_tensor(x, tf.float64))
            dataset_zipped = tf.data.Dataset.zip((dataset_we,dataset_se,dataset_gt))    
            datasets[c][t]=dataset_zipped
    # exit()

    # setup training params
    # Instantiate an optimizer.
    batch_size  = 16
    # get model
    raw_inputs,outputs = model_generation.build_Att_BiLSTM(batch_size=batch_size)
    model = keras.Model(raw_inputs, outputs)

    # run training loop
    print("Training this:")
    model.summary()
    keras.utils.plot_model(model,to_file="model_out.png",show_shapes=True,expand_nested=True)

    # Instantiate an optimizer.
    lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=1e-2,
        decay_steps=100,
        decay_rate=0.8, # sweeps from 0.01 to 0.0001 over the 10 epochs
        staircase=False)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    # Instantiate a loss function.
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)


    current_dir = "{:%m_%d_%H_%M}".format(datetime.now(timezone.utc))
    model_save_path = "/app/data/models/"+current_dir
    print("checkpoints stored in:\n\t{}".format(model_save_path))
    os.makedirs(model_save_path,exist_ok=True)

    # Prepare the metrics.
    train_acc_metric = keras.metrics.BinaryCrossentropy()
    val_acc_metric   = keras.metrics.BinaryCrossentropy()

    model.compile(run_eagerly=True)

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
        val_logits = model((we,se), training=False)
        val_acc_metric.update_state(gt, val_logits)


    epochs = 10
    for epoch in range(epochs):
        if epoch>0:
            model.save_weights(os.path.join(model_save_path,"model_epoch_{}".format(epoch)))
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (we_batch,se_batch,gt_batch) in enumerate(datasets["city"]["train"].batch(batch_size,drop_remainder=True)):
            loss_value = train_step(we_batch,se_batch,gt_batch)

            # Log every 10 batches.
            if step % 20 == 0:
                print("Training loss (for one batch) at step {}: {:.4}".format(step, float(loss_value)))
                print("Seen so far: %s samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: {:.4}".format(float(train_acc),))
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
        # Run a validation loop at the end of each epoch.
        for (we_batch,se_batch,gt_batch) in datasets["city"]["validation"].batch(batch_size,drop_remainder=True):
            test_step(we_batch,se_batch,gt_batch)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))


    # save model
    model.save_weights(os.path.join(model_save_path,"model_final"))