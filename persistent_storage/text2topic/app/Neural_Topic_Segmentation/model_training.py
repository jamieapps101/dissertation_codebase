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


# import from other files
import model_generation


def get_and_zip_data(files,dataset_dir):
    train_we_ds      = tf.data.TFRecordDataset([dataset_dir+files[0] for file_name in train_we_files]).map(lambda x: tf.io.parse_tensor(x, tf.float64))
    train_se_ds      = tf.data.TFRecordDataset([dataset_dir+files[1] for file_name in train_se_files]).map(lambda x: tf.io.parse_tensor(x, tf.float64))
    train_gt_ds      = tf.data.TFRecordDataset([dataset_dir+files[2] for file_name in train_gt_files]).map(lambda x: tf.io.parse_tensor(x, tf.float64))
    return tf.data.Dataset.zip((train_we_ds,train_se_ds,train_gt_ds))

if __name__=="__main__":
    # get model


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
    batch_size  = 64
    # get model
    raw_inputs,outputs = model_generation.build_Att_BiLSTM(batch_size=batch_size)
    model = keras.Model(raw_inputs, outputs)

    # run training loop
    print("Training this:")
    model.summary()
    keras.utils.plot_model(model,to_file="model_out.png",show_shapes=True,expand_nested=True)

    # Instantiate an optimizer.
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


    epochs = 1
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        # Iterate over the batches of the dataset.
        for step, (we_batch,se_batch,gt_batch) in enumerate(datasets["city"]["train"].batch(batch_size)):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                model_out = model((we_batch,se_batch), training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(gt_batch, model_out)
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))


    # save model
