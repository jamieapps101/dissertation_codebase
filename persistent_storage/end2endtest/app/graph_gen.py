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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_lc(event_file, scalar_str):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    lc = np.stack(
      [np.asarray([scalar.step, scalar.value])
      for scalar in ea.Scalars(scalar_str)])
    return(lc)

def do_save(fig,name):
    for ext in [".png",".eps"]:
        fn_name = os.path.join(GRAPH_OUT_DIR, name+ext)
        fig.savefig(fn_name,
            dpi=300)

def plot_tf_graphs():
    EPOCHS = 10
    # Pk graph
    fn_thresh = [(os.path.join(LOCAL_DATA_DIR,"run-04_20_14_14_Pk-tag-Pk_{:0.2f}.csv".format(thresh)),thresh) for thresh in np.arange(start=0.1,stop=1,step=0.1)]
    data = None
    for fn,thresh in fn_thresh:
        with open(fn, "r") as fp:
            local_data = pd.read_csv(fp)
        if data is None:
            data = local_data
            data = data.rename(columns={"Value":"Pk_{:0.2f}".format(thresh)})
        else:
            data["Pk_{:0.2f}".format(thresh)] = local_data["Value"]
    epoch = data["Step"]*EPOCHS/data["Step"].iloc[-1]
    data["Epoch"] = epoch
    
    y_labels = ["Pk_{:0.2f}".format(thresh) for thresh in np.arange(start=0.1,stop=1,step=0.1)]
    fig, ax = plt.subplots()
    data.plot(
        x="Epoch", y=y_labels,
        ax=ax,logy=True,
        ylabel="Pk (Unitless)",grid=True)
    plt.grid(True,"both",axis="y")
    do_save(fig,"Pk_graph")

    # Lr graph
    lr_data =  pd.read_csv(os.path.join(LOCAL_DATA_DIR,"run-04_20_14_14_lr-tag-Learn-Rate.csv"))
    epoch = lr_data["Step"]*EPOCHS/lr_data["Step"].iloc[-1]
    lr_data["Epoch"] = epoch
    fig, ax = plt.subplots()
    lr_data.plot(
        x="Epoch", y="Value",
        ax=ax, ylabel="Learning Rate (Unitless)",
        grid=True)
    do_save(fig,"Lr_graph")
    
    # Accuracy graph
    acc_data  = pd.read_csv(os.path.join(LOCAL_DATA_DIR,"run-04_20_14_14_test-tag-accuracy.csv"))
    acc_data = acc_data.rename(columns={"Value":"Test"})
    temp  = pd.read_csv(os.path.join(LOCAL_DATA_DIR,"run-04_20_14_14_train-tag-accuracy.csv"))
    acc_data["Train"] = temp["Value"]
    acc_data["Epoch"] = acc_data["Step"]*EPOCHS/acc_data["Step"].iloc[-1]
    fig, ax = plt.subplots()
    acc_data.plot(
        x="Epoch", y=["Test","Train"],
        ax=ax, ylabel="Accuracy (%)",
        grid=True)
    do_save(fig,"accuracy_graph")

    # Loss function graph
    loss_data  = pd.read_csv(os.path.join(LOCAL_DATA_DIR,"run-04_20_14_14_test-tag-loss.csv"))
    loss_data = loss_data.rename(columns={"Value":"Test"})
    temp  = pd.read_csv(os.path.join(LOCAL_DATA_DIR,"run-04_20_14_14_train-tag-loss.csv"))
    loss_data["Train"] = temp["Value"]
    loss_data["Epoch"] = loss_data["Step"]*EPOCHS/loss_data["Step"].iloc[-1]
    fig, ax = plt.subplots()
    loss_data.plot(
        x="Epoch", y=["Test","Train"],
        ax=ax, ylabel="Loss Value (Unitless)",
        grid=True)
    do_save(fig,"loss_graph")
    return loss_data

def plot_testing_data(config):
    pass


if __name__=="__main__":
    plot_tf_graphs()
    