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

END2END_DATA_DIR = "/app/results"
CONFIG = "/app/graphconfig.json"

import os
import tensorflow as tf
# from tensorflow.python.summary.summary_iterator import summary_iterator
# from tensorflow.python.summary import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import re
import seaborn as sns


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
    # load in data
    data= None
    for sample in config["data"]:
        data_local = None
        if sample["fn"]=="":
            continue
        with open(os.path.join(END2END_DATA_DIR,sample["fn"]), "r") as fp:
            data_local = pd.read_csv(fp)
        data_local["seg_thresh"] = np.ones((data_local.shape[0]))*sample["seg_thresh"]

        if data is None:
            data = data_local
        else:
            data = pd.concat((data,data_local),ignore_index=True)

    # get some data!
    ## get ave Pk for different segmentation thresholds
    # threshes = data["thresh_col"].unique()
    # for thresh in threshes:
        # data[data["thresh_col"]==thresh]
    
    # prog = re.compile("\d\.\d\d")
    # com_threshes = [float(re.findall(prog, col_name)[0]) for col_name in data.columns]
    # for com_thresh in com_threshes:
        # pass

    return data



        


if __name__=="__main__":
    # plot_tf_graphs()

    config=None
    with open(CONFIG, "r") as fp:
        config = json.load(fp)
    data = plot_testing_data(config)

# ['Unnamed: 0', 'sample_id', 'wer', 'pk', 'command_result', 'C', 'D', 'I',
#        'S', 'WER', 'thresh_0.700', 'thresh_0.710', 'thresh_0.720',
#        'thresh_0.730', 'thresh_0.740', 'thresh_0.750', 'thresh_0.760',
#        'thresh_0.770', 'thresh_0.780', 'thresh_0.790', 'thresh_0.800',
#        'thresh_0.810', 'thresh_0.820', 'thresh_0.830', 'thresh_0.840',
#        'thresh_0.850', 'thresh_0.860', 'thresh_0.870', 'thresh_0.880',
#        'thresh_0.890', 'thresh_0.900', 'thresh_0.910', 'thresh_0.920',
#        'thresh_0.930', 'thresh_0.940', 'thresh_0.950', 'thresh_0.960',
#        'thresh_0.970', 'thresh_0.980', 'thresh_0.990', 'thresh_1.000',
#        'seg_thresh'],

    ## get ave Pk for different segmentation thresholds
    sns_plot  = sns.relplot(x="seg_thresh", y="pk", kind="line", data=data); 
    do_save(sns_plot,"seg_thresh_vs_pk")



    ## get ave command accuracy for different command thresholds (conf matrix)
    prog = re.compile("\d\.\d\d")
    com_threshes = []
    col_names = []
    for col_name in data.columns:
        reg_matches=re.findall(prog, col_name)
        if len(reg_matches)>0:  
            com_threshes.append(float(reg_matches[0]))
            col_names.append(col_name)
    
    new_data = {
        "proportion":[],
        "classification":[],
        "threshold":[],
    }
    lookup = {
            "correct":   2,
            "incorrect": 1,
            "no_match":  0,
    }

    for col_name,thresh in zip(col_names,com_threshes):
        thresh_res = data[col_name].value_counts()
        thresh_acc = 100*thresh_res/thresh_res.sum()

        
        for key,val in lookup.items():
            new_data["classification"].append(key)
            new_data["proportion"].append(0 if val not in thresh_acc else thresh_acc[val])
            new_data["threshold"].append(thresh)
    new_data_df = pd.DataFrame.from_dict(new_data)
    

    sns_plot  = sns.relplot(x="threshold", y="proportion", hue="classification",
    kind="line", data=new_data); 
    do_save(sns_plot,"com_acc_vs_comm_thresh")

    # exit()
    # get best performing params:
    # for each segmentation thresh
        # need to get val counts for each of the different com thresh cols
            # get params with highest number of twos

            # get params for largest positive diff between twos and ones

    raw_perf_data = {
        "proportion":[],
        "classification":[],
        "seg_thresh":[],
        "com_thresh":[],
    }

    proc_perf_data = {
        "diff":[],
        "seg_thresh":[],
        "com_thresh":[],
    }
    for seg_thresh in data["seg_thresh"].unique():
            for col_name,com_thresh in zip(col_names,com_threshes):
                thresh_res = data[data["seg_thresh"]==seg_thresh][col_name].value_counts()
                thresh_acc = 100*thresh_res/thresh_res.sum()
                # if len(thresh_acc)>1:
                # print("seg_thresh: {}, com_thresh: {}\nres:{}\n".format(
                        # seg_thresh,com_thresh,thresh_acc))
                for key,val in lookup.items():
                    raw_perf_data["classification"].append(key)
                    raw_perf_data["proportion"].append(0 if val not in thresh_acc else thresh_acc[val])
                    raw_perf_data["com_thresh"].append(com_thresh)
                    raw_perf_data["seg_thresh"].append(seg_thresh)

                correct   = 0 if 2 not in thresh_acc else thresh_acc[2]
                incorrect = 0 if 1 not in thresh_acc else thresh_acc[1]
                diff = correct-incorrect # more positive is better
                proc_perf_data["diff"].append(diff)
                proc_perf_data["seg_thresh"].append(seg_thresh)
                proc_perf_data["com_thresh"].append(com_thresh)

    raw_perf_data_df = pd.DataFrame.from_dict(raw_perf_data)
    proc_perf_data_df = pd.DataFrame.from_dict(proc_perf_data)
    

    sns_plot = sns.JointGrid(
        data=proc_perf_data_df, 
        y="seg_thresh", x="com_thresh",space=0)

    sns_plot.plot_joint(sns.kdeplot,
        fill=True,
        cmap=sns.cubehelix_palette(start=.5, rot=-.5, reverse=True,as_cmap=True)
,
        legend=True,thresh=0)
    # sns_plot.plot_marginals(sns.histplot,color="#03051A", alpha=1, bins=25)

    do_save(sns_plot,"seg_thresh_vs_com_thresh_vs_diff")

    biggest_diff_index = proc_perf_data_df["diff"].argmax()
    best_result = proc_perf_data_df.iloc[biggest_diff_index]

    com_thresh = best_result["com_thresh"]
    seg_thresh = best_result["seg_thresh"]

    # ref this heavily
    raw_best_result = raw_perf_data_df.loc[
        np.logical_and(
            raw_perf_data_df["com_thresh"]==com_thresh,
            raw_perf_data_df["seg_thresh"]==seg_thresh
            )]

    # now get some summary metrics desribing the accruacy of each module
    
    # word error rate
    avg_wer = data["WER"].mean()
    std_wer = data["WER"].std()
    pc_std_wer = 100*std_wer/avg_wer
    print("avg_wer:{}".format(avg_wer))
    print("std_wer:{}".format(std_wer))
    print("pc_std_wer:{}".format(pc_std_wer))
    # plot distribution
    hist, bins = np.histogram(data["WER"])
    fig, ax = plt.subplots()
    ax.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]))
    ax.set_xlabel("WER")
    ax.set_ylabel("proportion of results")
    do_save(fig,"wer_dist")

    # pk
    avg_pk  = data["pk"].mean()
    std_pk = data["pk"].std()
    pc_std_pk = 100*std_pk/avg_pk
    print("avg_pk:{}".format(avg_pk))
    print("std_pk:{}".format(std_pk))
    print("pc_std_pk:{}".format(pc_std_pk))
    # plot distribution
    hist, bins = np.histogram(data["pk"])
    fig, ax = plt.subplots()
    ax.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]))
    ax.set_xlabel("Pk")
    ax.set_ylabel("proportion of results")
    do_save(fig,"pk_dist")

    # scatter graph of WER with Pk
    fig, ax = plt.subplots()
    data.plot.scatter(x="WER",y="pk",ax=ax)
    do_save(fig,"WER_pk_scatter")
    WER_pk_corr = data[["WER","pk"]].corr()
    print("WER_pk_corr:{}".format(WER_pk_corr))
    # given Pk for top WER results
    ## get data WER in the lowest 10%
    best_wer = data[data["WER"]<data["WER"].quantile(0.10)]

    # avg commnd comparison accuracy
    # avg_pk  = data["pk"].mean()
    # std_pk = data["pk"].std()
    # pc_std_pk = 100*std_pk/avg_pk
    # print("avg_pk:{}".format(avg_pk))
    # print("std_pk:{}".format(std_pk))
    # print("pc_std_pk:{}".format(pc_std_pk))
    sns_plot=sns.relplot(x="com_thresh", y="proportion",
             hue="seg_thresh", kind="line",
             data=raw_perf_data_df[raw_perf_data_df["classification"]=="correct"])
    do_save(sns_plot,"comm_thresh_vs_correct")

    sns_plot=sns.relplot(x="com_thresh", y="proportion",
             hue="seg_thresh", kind="line",
             data=raw_perf_data_df[raw_perf_data_df["classification"]=="incorrect"])
    do_save(sns_plot,"comm_thresh_vs_incorrect")


