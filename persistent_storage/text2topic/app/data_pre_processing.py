import json
import tensorflow as tf
import os
import numpy as np
import re
import pandas as pd
import threading
from multiprocessing import Process
from bert_serving.client import BertClient
import requests
import nltk.data
from nltk.tokenize import sent_tokenize,word_tokenize
import matplotlib.pyplot as plt


SAME_SAMPLES_AS_SOURCE = True


# supply array of sentences
def get_bert_encoding(data,bc):
    return bc.encode(data)

# supply array of array of words
def get_word2vec_encoding(word_data,max_sentence_len,max_sentences):
    # post request to word to vec container
    word_encoding_lenth = 300

    n_sentences = len(word_data)
    out = np.zeros((n_sentences,max_sentence_len,word_encoding_lenth))

    for sentence_index in range(min(max_sentences,len(word_data))):
        sentence = word_data[sentence_index]
        # send 
        response = requests.get("http://127.0.0.1:3030/convert/",data=json.dumps({"words": sentence}))
        response = json.loads(response.content)

        if len(sentence)>max_sentence_len:
            # print("too many words ({}/{})".format(len(sentence),max_sentence_len))
            print("w",end="")
            return None

        for word_index in range(len(sentence)):
            word_text = sentence[word_index]
            response_data = response["data"][word_text]
            if response_data is not None:
                out[sentence_index,word_index,:] = np.array(response_data)

    return out


data_info_csv_name="data_info.csv"
data_dir = "/app/data/"
dataset_dir = "/app/data/processed"
raw_ds_fn = ["wikisection_en_","_",".json"]

discordant_pairs = 100



# mode = "inspect"
mode = "extract"

def analyse(raw_data_path, data_info_csv_path,c_id,d_id):
    data = None
    data_info = pd.DataFrame(data={"content":[],"dataset":[],"sample_index":[],"sentence_index":[],"sentence_length":[]})
    with open(raw_data_path,"r") as fp:
        print("successfully opened {}".format(fn))
        data = json.load(fp)
    if data is None:
        print("error loading {}, skipping file".format(fn))
        return
    available_samples = len(data)
    # this mode determines the longest sentence in the set
    #for each sample
    word_exp = re.compile("[a-zA-Z\-]+")
    sentence_split_exp = re.compile("[.\\n]")

    next_print_thresh = 0.0

    print("available_samples = {}".format(available_samples))
    for i in range(available_samples): 
        if i/available_samples > next_print_thresh:
            print("-> {}% of {}".format(next_print_thresh*100,raw_data_path))
            next_print_thresh = next_print_thresh+0.01
        # split text into sentences
        S = sentence_split_exp.split(data[i]["text"])
        for s_id in range(len(S)):
            # s = S[s_id]
            current_sentence_len = len(word_exp.findall(S[s_id]))
            data_point = {
                "content":c_id,
                "dataset":d_id,
                "sample_index":i,
                "sentence_index":s_id,
                "sentence_length":current_sentence_len
                }
            data_info = data_info.append(data_point,ignore_index=True)
    with open(data_info_csv_path,"w") as fp:
        data_info.to_csv(fp)


# using tf.train.Example
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# using tf.train.Example
def serialize_example(se, we, gt):
  feature = {
      'se': _bytes_feature(se),
      'we': _bytes_feature(we),
      'gt': _bytes_feature(gt),
  }  # Create a Features message using tf.train.Example.  
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def serialize_example2(se, we, gt):
  feature = {
      'se': _bytes_feature(tf.io.serialize_tensor(se)),
      'we': _bytes_feature(tf.io.serialize_tensor(we)),
      'gt': _bytes_feature(tf.io.serialize_tensor(gt)),
  }  # Create a Features message using tf.train.Example.  
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def tf_serialize_example(we, se, gt):
  tf_string = tf.py_function(
    serialize_example2,
    (se, we, gt),  # pass these args to the above function.
    tf.string)      # the return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar

def write_datasets_zipped2(data_we,data_se,data_gt,fn):
    combined_data = tf.data.Dataset.from_tensor_slices((data_we,data_se,data_gt))
    # wrap function in tf freindly wrapper
    serialized_dataset = combined_data.map(tf_serialize_example)
    path   = os.path.join(dataset_dir, fn)
    writer =  tf.data.experimental.TFRecordWriter(path)
    writer.write(serialized_dataset)

def write_datasets_zipped(data_we,data_se,data_gt,fn):
    # zip datasets togeather
    zipped = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(data_we),
            tf.data.Dataset.from_tensor_slices(data_se),
            tf.data.Dataset.from_tensor_slices(data_gt)
        )
    )

    # mapped_data = zipped.map(write_map_fn)
    # fn     = content[c_index]+"_"+datasets[d_index]+"_{}.tfrecord".format(buffers_written)
    path   = os.path.join(dataset_dir, fn)
    with tf.io.TFRecordWriter(path) as writer:
        rows = data_se.shape[0]
        # for row in range(rows):
        for we,se,gt in zipped:
            serialised_data = serialize_example(
                tf.io.serialize_tensor(se), 
                tf.io.serialize_tensor(we), 
                tf.io.serialize_tensor(gt)
                # data_se[row], 
                # data_we[row], 
                # data_gt[row]
                )
            writer.write(serialised_data)



def write_datasets(data_gt,buffers_written,ext):
    fn_gt = content[c_index]+"_"+datasets[d_index]+"_"+ext+"_{}.tfrecord".format(buffers_written)
    path_gt = os.path.join(dataset_dir, fn_gt)
    data_gt_t = tf.data.Dataset.from_tensor_slices(data_gt)

    # main_dataset = tf.data.Dataset.zip((data_we_t, data_se_t, data_gt_t))


    def write_map_fn(inlet):
        return tf.io.serialize_tensor(inlet)
    
    # mapped_dataset = main_dataset.map(write_map_fn)
    mapped_data_gt_t = data_gt_t.map(write_map_fn)

    writer_gt = tf.data.experimental.TFRecordWriter(path_gt)
    
    writer_gt.write(mapped_data_gt_t)


if __name__=="__main__":
    content  = ["disease","city"]
    datasets = ["validation","test","train"]

    filenames = []
    for c in content:
        for d in datasets:
            filenames.append(raw_ds_fn[0]+c+raw_ds_fn[1]+d+raw_ds_fn[2])

             
    if mode=="extract":
        print("extract mode")
        # connect to bert client
        # bc = BertClient(ip="jamie-laptop") # this defo wants fixing
        bc = BertClient(ip="127.0.0.1") # this defo wants fixing
        max_sentence_len = 90
        max_sentences_per_sample = 100 # this is for both segments
        word2vec_encoding_len = 300
        bert_encoding_len = 1024
        max_buffer_length = 250


        word_exp = re.compile("[a-zA-Z\-]+")
        sentence_split_exp = re.compile("[.\\n]")

        samples_per_set = [
            [100,200,700], # disease
            [100,200,700], # city
        ]

        print("beginning")
        for c_index in range(len(content)):
            for d_index in range(len(datasets)):
                print("c_index={}, d_index={}".format(c_index,d_index))
                fn = raw_ds_fn[0]+content[c_index]+raw_ds_fn[1]+datasets[d_index]+raw_ds_fn[2]
                # read in file
                raw_data_path = os.path.join(data_dir,"raw/WikiSection-master",fn)
                data = None
                with open(raw_data_path,"r") as fp:
                    print("successfully opened {}".format(fn))
                    data = json.load(fp)
                if data is None:
                    print("error loading {}, skipping file".format(fn))
                    continue
                print("data loaded")

                available_samples = len(data)
                print("available_samples: {}".format(available_samples))
                if SAME_SAMPLES_AS_SOURCE:
                    samples_per_set[c_index][d_index]=available_samples
                

                actual_buffer_length = max_buffer_length 
                if max_buffer_length>samples_per_set[c_index][d_index]:
                    actual_buffer_length=samples_per_set[c_index][d_index]

                print("using buffer length: {}".format(actual_buffer_length))

                print("allocating buffer ram")
                # create a data record so we know which samples we have combined before
                index_record = np.array([[-1,-1]])
                # [sample,sentence index,word index, embedding length]
                data_we = np.zeros([actual_buffer_length,max_sentences_per_sample,max_sentence_len,word2vec_encoding_len])
                # [sample,sentence index, embedding length]
                data_se = np.zeros([actual_buffer_length,max_sentences_per_sample,bert_encoding_len]) 
                # [sample,sentence index, 1], where 1 is a bool indicating topic change
                data_gt = np.zeros([actual_buffer_length,max_sentences_per_sample,1])

                samples_in_buffer = 0
                buffers_written = 0

                print("beginning sample generation")
                # loop and keep on looping until the required number of samples
                rand_indexes      = np.array([[0,1]])
                prev_rand_indexes = np.array([[0,0]])
                last_update = 0
                while True:
                    # give indication of progress
                    if index_record.shape[0]%10 == 0 and last_update!=index_record.shape[0]:
                        last_update=index_record.shape[0]
                        print("\ngenerated {}/{} records".format(index_record.shape[0],samples_per_set[c_index][d_index]))

                    
                    # gen some random indexes
                    prev_rand_indexes = rand_indexes
                    while True:
                        rand_indexes = np.random.randint(available_samples,size=(1,2))
                        if not np.any(np.all(index_record == rand_indexes,axis=1)):
                            break
   
                    # process first sample
                    ## get sentences
                    sentences = sent_tokenize(data[rand_indexes[0,0]]["text"])
                    ## get word to vec encodings
                    word2vec_encodings = get_word2vec_encoding(
                        [[w for w in word_tokenize(s) if word_exp.search(w) is not None] for s in sentences if len(s)>0],
                        max_sentence_len,max_sentences_per_sample)

                    # skip this round if this sample has too many words in a sentence
                    if word2vec_encodings is None:
                        continue
                    current_sentence_len = word2vec_encodings.shape[0]
                    if max_sentences_per_sample<word2vec_encodings.shape[0]:
                        continue

                    # get bert encodings
                    bert_encodings = get_bert_encoding(sentences,bc)

                    # send to buffer
                    data_we[samples_in_buffer,0:current_sentence_len,:,:] = word2vec_encodings
                    data_se[samples_in_buffer,0:current_sentence_len,:]   = bert_encodings[:current_sentence_len]
                    data_gt[samples_in_buffer,current_sentence_len-1] = 1
                    # account for fact that previous samples will affect current lstm state
                    # therefore add a boundry indicator to the start of the sample
                    if rand_indexes[0,0] != prev_rand_indexes[0,1]:
                        data_gt[samples_in_buffer,0] = 1
                    # process second sample
                    ## get sentences
                    sentences = sent_tokenize(data[rand_indexes[0,1]]["text"])
                    ## get word to vec encodings
                    word2vec_encodings = get_word2vec_encoding(
                        [[w for w in word_tokenize(s) if word_exp.search(w) is not None] for s in sentences if len(s)>0],
                        max_sentence_len,max_sentences_per_sample)

                    # skip this round if this sample has too many words in a sentence
                    if word2vec_encodings is None:
                        continue

                    additional_sentences = word2vec_encodings.shape[0]
                    if current_sentence_len+additional_sentences < max_sentences_per_sample:
                        final_abs_sentence_index = current_sentence_len+additional_sentences
                        final_rel_sentence_index = additional_sentences
                    else:
                        final_abs_sentence_index = max_sentences_per_sample
                        final_rel_sentence_index = max_sentences_per_sample-current_sentence_len

                    # get bert encodings
                    bert_encodings = get_bert_encoding(sentences,bc)

                    # send to buffer
                    data_we[samples_in_buffer,current_sentence_len:final_abs_sentence_index,:,:] = word2vec_encodings[:final_rel_sentence_index]
                    data_se[samples_in_buffer,current_sentence_len:final_abs_sentence_index,:]   = bert_encodings[:final_rel_sentence_index]


                    # if we've gotten to here, the data must be ok, so add it to the record
                    samples_in_buffer+=1
                    if np.all(index_record==np.array([[-1,-1]])):
                        index_record = rand_indexes
                    else:   
                        index_record = np.append(index_record,rand_indexes,axis=0)

                    if samples_in_buffer==actual_buffer_length:
                        print("writing data!")
                        fn = content[c_index]+"_"+datasets[d_index]+"_{}.tfrecord".format(buffers_written)
                        print("writing batch number {} to file:\n\t{}".format(buffers_written,fn))
                        # slices required here as we don't want any zero padding being carried into the training data
                        write_datasets_zipped2(data_we[:samples_in_buffer],data_se[:samples_in_buffer],data_gt[:samples_in_buffer],fn)

                        # reset data valirables
                        samples_in_buffer = 0
                        buffers_written += 1
                        # reset buffers just to be sure
                        data_we = np.zeros([actual_buffer_length,max_sentences_per_sample,max_sentence_len,word2vec_encoding_len])
                        data_se = np.zeros([actual_buffer_length,max_sentences_per_sample,bert_encoding_len]) 
                        data_gt = np.zeros([actual_buffer_length,max_sentences_per_sample,1])
                    if index_record.shape[0] >= samples_per_set[c_index][d_index]:
                        # write out record of which sentences were used
                        fn = content[c_index]+"_"+datasets[d_index]+"_{}.csv".format(buffers_written)
                        path   = os.path.join(dataset_dir, fn)
                        np.savetxt(path,index_record,delimiter=',')
                        print("exiting loop")
                        break

                if samples_in_buffer>0:
                    print("Final batch of data")
                    fn = content[c_index]+"_"+datasets[d_index]+"_{}.tfrecord".format(buffers_written)
                    print("writing batch number {} to file:\n\t{}".format(buffers_written,fn))
                    # slices required here as we don't want any zero padding being carried into the training data
                    write_datasets_zipped2(data_we[:samples_in_buffer],data_se[:samples_in_buffer],data_gt[:samples_in_buffer],fn)
                    # reset data valirables
                    samples_in_buffer = 0
                    buffers_written += 1



    if mode=="inspect":
        # for test,train,validation
        max_sentence_len = 0
        min_sentence_len = 100
        c_id = 0
        d_id = -1
        thread_pool = []
        for fn in filenames:
            d_id = d_id+1
            if d_id == len(datasets):
                c_id = c_id+1
                d_id = 0
            # check if file already analysed
            data_info_csv_path = os.path.join(data_dir,"metrics","{}-{}-".format(datasets[d_id],content[c_id])+data_info_csv_name)
            if (not os.path.exists(data_info_csv_path)) or False: # a switch just to regen data if needed
                # if not, setup a thread to begin anaylysing
                raw_data_path = os.path.join(data_dir,"raw/WikiSection-master",fn)
                temp = Process(target=analyse, args=(raw_data_path, data_info_csv_path,c_id,d_id))
                thread_pool.append(temp)
                temp.start()
            
        if len(thread_pool)>0:
            print("waiting for threads to finish")
            for t in thread_pool:
                t.join()                 
        
        data = []
        c_id = 0
        d_id = -1
        print("Reading in saved data")
        for fn in filenames:
            d_id = d_id+1
            if d_id == len(datasets):
                c_id = c_id+1
                d_id = 0
            data_info_csv_path = os.path.join(data_dir,"metrics","{}-{}-".format(datasets[d_id],content[c_id])+data_info_csv_name)
            with open(data_info_csv_path,"r") as fp:
                data.append(pd.read_csv(fp))
        data_info = pd.concat(data,ignore_index=True)
        # now do some analysis
        max_sentence_id = data_info["sentence_length"].argmax()
        max_sentence_record = data_info.iloc[max_sentence_id]
        print("max_sentence_record:\n{}".format(max_sentence_record))
        min_sentence_id = data_info["sentence_length"].argmin()
        min_sentence_record = data_info.iloc[min_sentence_id]
        print("min_sentence_record:\n{}".format(min_sentence_record))
        description = data_info["sentence_length"].describe()
        print("general description:\n{}".format(description))

        # max length index
        # content: 1, dataset: 2, sample_index=690,sentence_index=306,sentence_length: 293
        # city,train
        fig, ax = plt.subplots(1,2,figsize=(20, 10),dpi=80)

        data_info.hist(column="sentence_length",ax=ax[0],grid=True,bins=20,log=True)
        ax[0].set_title('Distribution of Sentence Lengths')
        ax[0].set_xlabel("Sentence Length (words)")
        ax[0].set_ylabel("Frequency")


        # "content":[],"dataset":[],"sample_index":[],"sentence_index":[],"sentence_length":

        sentence_lengths = data_info.groupby(["content","dataset","sample_index"]).agg({'sentence_index': 'max'})


        sentence_lengths.hist(column="sentence_index",ax=ax[1],grid=True,bins=20,log=True)
        ax[1].set_title('Distribution of Sentences Per Sample')
        ax[1].set_xlabel("Sample Length (Sentences)")
        ax[1].set_ylabel("Frequency")

        fig.savefig("distribution.png")