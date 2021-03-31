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


# supply array of sentences
def get_bert_encoding(data,bc,max_sentences,bert_encoding_len):
    # post request to bert as a service
    # for each sentence
    output = np.array([max_sentences,bert_encoding_len])
    
    output = bc.encode(data)
    # return data response
    return output

# supply array of array of words
def get_word2vec_encoding(word_data,max_sentence_len,max_sentences):
    # post request to word to vec container
    word_encoding_lenth = 300
    # using zero for padding
    out = np.zeros((max_sentences,max_sentence_len,word_encoding_lenth))

    if len(word_data) > max_sentences:
        # print("too many sentences ({}/{})".format(len(word_data),max_sentences))
        return None

    for sentence_index in range(len(word_data)):
        sentence = word_data[sentence_index]
        # send 
        response = requests.get("http://127.0.0.1:3030/convert/",data=json.dumps({"words": sentence}))
        response = json.loads(response.content)

        if len(sentence)>max_sentence_len:
            # print("too many words ({}/{})".format(len(sentence),max_sentence_len))
            return None

        for word_index in range(len(sentence)):
            word_text = sentence[word_index]
            response_data = response["data"][word_text]
            
            if response_data is not None:
                out[sentence_index,word_index,:] = np.array(response_data)
                # print("no entry for {}".format(word_text))
                # have a list of common words that we'll ignore for now
                # if word_text not in ["of"]:
                    # missing_words = missing_words+1
                    #  return None # cannot use this sample as it has an unrecognised word

    # print("missing_words: {}".format(missing_words))
    # return data response
    # print("good sample")
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


def sample_to_frame(sample,max_sentence_len,bc,bert_encoding_len=1024):
    # split sample into sentences
    word_exp = re.compile("[a-zA-Z\-'`]+")
    sentence_split_exp = re.compile("[a-z0-9]\.[\s\\n]+[A-Z0-9]")
    S = sentence_split_exp.split(sample["text"])

    # first encode the sentences with bert
    bert_encodings = get_bert_encoding(S,bc)
    # word2vec_encodings = get_word2vec_encoding([word_exp.findall(s) for s in sentence_split_exp.split(S) if len(s)>0],max_sentence_len)

    return word2vec_encodings,bert_encodings


def write_datasets_zipped(data_we,data_se,data_gt,fn):
    # zip datasets togeather
    zipped = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(data_we),
            tf.data.Dataset.from_tensor_slices(data_se),
            tf.data.Dataset.from_tensor_slices(data_gt)
            )
    )

    def write_map_fn(x1, x2, x3):
        return tf.io.serialize_tensor(x1)+tf.io.serialize_tensor(x2)+tf.io.serialize_tensor(x3)
        # return tf.io.serialize_tensor(inlet)
    mapped_data = zipped.map(write_map_fn)
    # fn     = content[c_index]+"_"+datasets[d_index]+"_{}.tfrecord".format(buffers_written)
    path   = os.path.join(dataset_dir, fn)
    writer = tf.data.experimental.TFRecordWriter(path)
    writer.write(mapped_data)



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

             
    if mode=="extract":
        print("extract mode")
        # connect to bert client
        # bc = BertClient(ip="jamie-laptop") # this defo wants fixing
        bc = BertClient(ip="127.0.0.1") # this defo wants fixing
        max_sentence_len = 80
        max_sentences_per_sample = 80 # this is for both segments
        word2vec_encoding_len = 300
        bert_encoding_len = 1024

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


                print("Allocating buffer ram")
                max_buffer_length = 1000

                actual_buffer_length = max_buffer_length 
                if max_buffer_length>samples_per_set[c_index][d_index]:
                    actual_buffer_length=samples_per_set[c_index][d_index]

                print("using buffer length: {}".format(actual_buffer_length))

                # create a data record so we know which samples we have combined before
                record = np.array([[-1,-1]])

                # [sample,sentence index,word index, embedding length]
                data_we = np.zeros([
                    # samples_per_set[c_index][d_index], 
                    actual_buffer_length,
                    max_sentences_per_sample,
                    max_sentence_len,
                    word2vec_encoding_len,
                    ])
                # [sample,sentence index, embedding length]
                data_se = np.zeros([
                    # samples_per_set[c_index][d_index], 
                    actual_buffer_length,
                    max_sentences_per_sample,
                    bert_encoding_len
                ]) 
                # [sample,sentence index, 1], where 1 is a bool indicating topic change
                data_gt = np.zeros([
                    # samples_per_set[c_index][d_index], 
                    actual_buffer_length,
                    max_sentences_per_sample,
                    1
                ])

                samples_in_buffer = 0
                buffers_written = 0

                samples = len(data)
                print("max samples: {}".format(samples))
                print("beginning sample generation")
                # loop and keep on looping until the required number of samples
                total_records = record.shape[0]
                while total_records<samples_per_set[c_index][d_index]:
                    if total_records%50 == 0:
                        print("generated {} records".format(total_records))
                    # gen some random indexes
                    while True:
                        rand_indexes = np.random.randint(samples,size=(2))
                        if not np.any(record == rand_indexes):
                            break
                    # for each sample, get sentences encodings and word encodings
                    previous_sentences = 0
                    bad_sample = False
                    remaining_sentences = max_sentences_per_sample
                    for index in [0,1]:
                        # split sample into sentences
                        word_exp = re.compile("[a-zA-Z\-'`]+")
                        sentence_split_exp = re.compile("([a-z0-9])\.[\s\\n]+[A-Z0-9]")
                        # sentences = sentence_split_exp.split(data[index]["text"])
                        sentences = sent_tokenize(data[rand_indexes[index]]["text"])

                        # first encode the sentences with bert
                        # word2vec_encodings = get_word2vec_encoding([word_exp.findall(s) for s in sentences if len(s)>0],max_sentence_len,max_sentences_per_sample)
                        word_regex = re.compile("[a-zA-Z]+")
                        word2vec_encodings = get_word2vec_encoding([[w for w in word_tokenize(s) if word_regex.search(w) is not None] for s in sentences if len(s)>0],max_sentence_len,remaining_sentences)
                        # exit()
                        if word2vec_encodings is None:
                            bad_sample = True
                            break
                        bert_encodings = get_bert_encoding(sentences,bc,max_sentences_per_sample,bert_encoding_len)
                        # add data into buffer
                        # if this is the first index
                        if index == 0: 
                            data_we[samples_in_buffer,0:len(sentences),:,:] = word2vec_encodings[0:len(sentences),:,:]
                            data_se[samples_in_buffer,0:len(sentences),:] = bert_encodings
                            previous_sentences = len(sentences)
                            remaining_sentences -= len(sentences)
                        else:
                        # if this is the second index
                            data_we[samples_in_buffer,previous_sentences:previous_sentences+len(sentences),:,:] = word2vec_encodings[0:len(sentences),:,:]
                            data_se[samples_in_buffer,previous_sentences:previous_sentences+len(sentences),:] = bert_encodings
                            # set first sentence here to have a 1 label
                            if rand_indexes[0] != rand_indexes[1]:
                                data_gt[samples_in_buffer,previous_sentences,0]=1
                            else:
                                print("same record found!")
                            previous_sentences = 0

                    if not bad_sample:
                        record = np.append(record,rand_indexes)
                        samples_in_buffer += 1                    
                        if samples_in_buffer >= max_buffer_length:
                            print("writing data!")
                            # TODO write out the data here
                            
                            fn = content[c_index]+"_"+datasets[d_index]+"_{}.tfrecord".format(buffers_written)
                            write_datasets_zipped(data_we,data_se,data_gt,fn)
                            # write_datasets(data_we,buffers_written,"we")
                            # write_datasets(data_se,buffers_written,"se")
                            # write_datasets(data_gt,buffers_written,"gt")
                            # reset data valirables
                            samples_in_buffer = 0
                            buffers_written += 1
                            # exit()
                    total_records = record.shape[0]
                if samples_in_buffer>0:
                    fn = content[c_index]+"_"+datasets[d_index]+"_{}.tfrecord".format(buffers_written)
                    write_datasets_zipped(data_we,data_se,data_gt,fn)
                    # write out any remining records after looping
                    # write_datasets(data_we,buffers_written,"we")
                    # write_datasets(data_se,buffers_written,"se")
                    # write_datasets(data_gt,buffers_written,"gt")

                    


                    



    if mode=="extract" and False:
        # connect to bert client
        bc = BertClient(ip="jamie-laptop") # this defo wants fixing
        max_sentence_len = 60
        word2vec_encoding_len = 300
        bert_encoding_len = 1024


        word_exp = re.compile("[a-zA-Z\-]+")
        sentence_split_exp = re.compile("[.\\n]")

        c_id = 0
        d_id = -1
        for fn in filenames:
            d_id = d_id+1
            if d_id == len(datasets):
                c_id = c_id+1
                d_id = 0
            # print("c_id:{} - d_id:{}".format(c_id,d_id))
            # check if file already analysed
            data_info_csv_path = os.path.join(data_dir,"metrics","{}-{}-".format(datasets[d_id],content[c_id])+data_info_csv_name)
            if not os.path.exists(data_info_csv_path):
                print("{} not found, skipping".format(data_info_csv_path))
            
            # print("a")

            # read in the file
            data = None
            raw_data_path = os.path.join(data_dir,"raw/WikiSection-master",fn)
            with open(raw_data_path,"r") as fp:
                print("successfully opened {}".format(fn))
                data = json.load(fp)
            if data is None:
                print("error parsing {}, skipping file".format(fn))
                continue

            # print("b")
            # make directory structure
            out_path = os.path.join(data_dir,"processed",datasets[d_id],content[c_id])
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            
            # begin converting the samples
            available_samples = len(data)


            # now begin looping
            for file_index in range(1): # 5
                for batch_index in range(1): # 10
                    samples_per_batch    = 20
                    topics_per_sequence  = 4
                    sentences_per_sample = 10
                    max_sentence_len     = 60

                    bert_vecs       = np.zeros(shape=(samples_per_batch,sentences_per_sample,bert_encoding_len))
                    word2vec_frames = np.zeros(shape=(samples_per_batch,sentences_per_sample,max_sentence_len,word2vec_encoding_len))
                    # generate a mask, this indicates how many topics are included in one sentence sequence
                    # this method is chosen as it garentees at least 1 topic per sample
                    mask_raw = np.random.randint(1,2**topics_per_sequence-1, size=(samples_per_batch,1,1),dtype="uint8")
                    mask = np.unpackbits(mask_raw,axis=2,count=4,bitorder="little")
                    
                    partition_proportion = np.random.rand(samples_per_batch,1,topics_per_sequence)

                    masked_partition_proportion = np.multiply(partition_proportion,mask)

                    partition = np.round(masked_partition_proportion/masked_partition_proportion.sum(axis=2,keepdims=True)*sentences_per_sample)
                    position  = np.round(masked_partition_proportion.cumsum(axis=2)/masked_partition_proportion.sum(axis=2,keepdims=2)*sentences_per_sample) 
                    topics    = np.random.randint(available_samples,size=(samples_per_batch,1,topics_per_sequence))

                    # print("partition: \n{}\n".format(partition))
                    # print("position: \n{}\n".format(position))
                    # print("topics: \n{}\n".format(topics))


                    for s_id in range(samples_per_batch):
                        for index,topic_id,topic_partition,topic_position in zip(range(topics_per_sequence),topics[s_id][0],partition[s_id][0],position[s_id][0]):
                            # get the topic text,
                            # convert to sentences
                            # S = sentence_split_exp.split(data[topic_id]["text"])
                            word2vec_encodings,bert_encodings = sample_to_frame(data[topic_id],max_sentence_len,bc,1024)

                            # bert_vecs[s_id][]





            # # allocate the space for the sample data

            # print("available_samples = {}".format(available_samples))
            # for i in range(available_samples): 
            #     if i/available_samples > next_print_thresh:
            #         print("-> {}% of {}".format(next_print_thresh*100,raw_data_path))
            #         next_print_thresh = next_print_thresh+0.01
            #     word2vec_encodings,bert_encodings = sample_to_frame(data[i],max_sentence_len,bc,1024)
            #     word2vec_frames[i,:,:,:]  = word2vec_encodings
            #     bert_vecs[i,:,:]          = bert_encodings

                
                

        # create discordant pairs
        # indexes = None
        # for i in range(discordant_pairs):
        #     indexes = np.random.randint(0,available_samples,size=(2))
        

    # create word2vect encoding

    # create bert encoding

    # add to numpy array


