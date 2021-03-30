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



if __name__=="__main__":
    # get model
    raw_inputs,outputs = model_generation.build_Att_BiLSTM()
    model = keras.Model(raw_inputs, outputs)


    # load data
    dataset_dir = "/app/data/processed"
    files = os.listdir(dataset_dir)
    train_files = [file_name for file_name in files if "train" in file_name]
    test_files = [file_name for file_name in files if "test" in file_name]
    validation_files = [file_name for file_name in files if "validation" in file_name]

    # overide this for testing
    train_we_files         = ["disease_train_we_0.tfrecord"]
    train_se_files         = ["disease_train_se_0.tfrecord"]
    train_gt_files         = ["disease_train_gt_0.tfrecord"]
    test_we_files          = ["disease_test_we_0.tfrecord"]
    test_se_files          = ["disease_test_se_0.tfrecord"]
    test_gt_files          = ["disease_test_gt_0.tfrecord"]
    validation_we_files    = ["disease_validation_we_0.tfrecord"]
    validation_se_files    = ["disease_validation_se_0.tfrecord"]
    validation_gt_files    = ["disease_validation_gt_0.tfrecord"]

    train_we_ds         = tf.data.TFRecordDataset(train_we_files)
    train_se_ds         = tf.data.TFRecordDataset(train_se_files)
    train_gt_ds         = tf.data.TFRecordDataset(train_gt_files)
    test_we_ds          = tf.data.TFRecordDataset(test_we_files)
    test_se_ds          = tf.data.TFRecordDataset(test_se_files)
    test_gt_ds          = tf.data.TFRecordDataset(test_gt_files)
    validation_we_ds        = tf.data.TFRecordDataset(validation_we_files)
    validation_se_ds        = tf.data.TFRecordDataset(validation_se_files)
    validation_gt_ds        = tf.data.TFRecordDataset(validation_gt_files)

    # setup training params
    # Instantiate an optimizer.
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    batch_size  = 64

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.zip((train_we_ds,train_se_ds,train_gt_ds))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.zip((validation_we_ds,validation_se_ds,validation_gt_ds))
    val_dataset = val_dataset.batch(batch_size)

    # run training loop

    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        # Iterate over the batches of the dataset.
        for step, (we_batch,se_batch,gt_batch) in enumerate(train_dataset):
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
