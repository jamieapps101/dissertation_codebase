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


def read_tfrecord(serialized_example):
  feature_description = {
        'se': tf.io.FixedLenFeature((), tf.string),
        'we': tf.io.FixedLenFeature((), tf.string),
        'gt': tf.io.FixedLenFeature((), tf.string)
  }
  example = tf.io.parse_single_example(serialized_example, feature_description)
  se = tf.io.parse_tensor(example['se'], out_type = tf.float64)
  we = tf.io.parse_tensor(example['we'], out_type = tf.float64)
  gt = tf.io.parse_tensor(example['gt'], out_type = tf.float64)
  return se, we, gt


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


    content = ["disease","city"]
    types   = ["train","test","validation"]
    items = [7,2,1]
    datasets = {}
    temp = None
    for c in content:
        datasets[c] = {}
        for shards,t in zip(items,types):
            files = []
            for i in range(shards):
                files_we.append("{}_{}_{}.tfrecord".format(c,t,i))
            dataset = tf.data.TFRecordDataset([dataset_dir+file_name for file_name in files_we]).map(read_tfrecord)

            # dataset_zipped = tf.data.Dataset.zip((dataset_we,dataset_se,dataset_gt))    
            # datasets[c][t]=dataset_zipped
            datasets[c][t]=(dataset_we,dataset_se,dataset_gt)
    exit()

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




    if True:
        # train using pre-made functions
        model.compile(
            optimizer=keras.optimizers.Adam(),
            # Loss function to minimize
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            # List of metrics to monitor
            metrics=[keras.metrics.SparseCategoricalAccuracy()]
        )

        # Train for 5 epochs with batch size of 32.
        print("Fit model on training data")
        history = model.fit(
            {"WE":datasets["city"]["train"][0],"SE":datasets["city"]["train"][1]},
            {"boundry_out":datasets["city"]["train"][2]},
            batch_size=16,
            epochs=2,
            steps_per_epoch=20,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data={"WE":datasets["city"]["validation"][0],"SE":datasets["city"]["validation"][1],"boundry_out":datasets["city"]["validation"][2]},
        )



    elif False:
        # Train model with manual trianing loops
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