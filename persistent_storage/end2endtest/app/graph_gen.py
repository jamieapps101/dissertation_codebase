# To do:
# tensorflow learning graphs
# - learning rate
# - Pk (stacked)
# - accuracy
# - loss function

# end to end testing graphs
# accuracy for 
# topic segmentation thres vs command match thresh

# have a stacked graph, each line being a topic segmentation thresh. 
# x axis being command match thresh, y axis being accuracy. end to end accuracy

LOCAL_DATA_DIR = "/app/graph_data"
TF_DATA_DIR    = "/text2topic/app/data/logs/04_20_14_14"
GRAPH_OUT_DIR  = "/app/graphs_out"


import os
import tensorflow as tf
# from tensorflow.python.summary.summary_iterator import summary_iterator
# from tensorflow.python.summary import event_accumulator
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd


def get_lc(event_file, scalar_str):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    lc = np.stack(
      [np.asarray([scalar.step, scalar.value])
      for scalar in ea.Scalars(scalar_str)])
    return(lc)

def plot_tf_graphs():
    # LR graph
    fn_thresh = [(os.path.join(LOCAL_DATA_DIR,"run-04_20_14_14_Pk-tag-Pk_{:0.2f}.csv".format(thresh)),thresh) for thresh in np.arange(start=0,stop=1,step=0.1)]
    data = None
    for fn,thresh in fn_thresh:
        with open(fn, "r") as fp:
            local_data = pd.read_csv(fp)
        if data is None:
            data = local_data
            data = data.rename(columns={"Value":"Pk_{:0.2f}".format(thresh)})
        else:
            data["Pk_{:0.2f}".format(thresh)] = local_data["Value"]

    y_labels = ["Pk_{:0.2f}".format(thresh) for thresh in np.arange(start=0,stop=1,step=0.1)]
    data.plot(x="Step", y=y_labels)
    return data        

    # Pk graph
    # Accuracy graph
    # Loss function graph

def plot_testing_data():
    pass


if __name__=="__main__":
    data = plot_tf_graphs()