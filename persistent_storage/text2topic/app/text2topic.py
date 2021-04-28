import model_generation
import model_inference
import model_training
import data_pre_processing

from bert_serving.client import BertClient
import json,os,re,time
import paho.mqtt.client as mqtt
import threading, queue
import pandas as pd
import numpy as np
import tensorflow as tf

SEG_THRESH = 0.0001


CONFIG_PATH = "/app/testconfig.json"
DATA_PATH   = "/app/data/processed"

BERT_IP = "10.80.0.2"
MQTT_HOST   = "127.0.0.1"
MQTT_PORT   = 1883

INPUT_TOPIC = "TS_input"
OUTPUT_TOPIC = "TS_output"

BUFFER_LEN = 10 

# setup queues to enable callbacks to inform main thread
TS_segments    = queue.Queue()

# setup MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # client.subscribe("$SYS/#")
    client.subscribe(INPUT_TOPIC)

def on_message(client, userdata, msg):
    global TS_segments
    print("recived text")
    if   msg.topic == INPUT_TOPIC:
        byte_string = msg.payload
        json_string = byte_string.decode("utf-8")
        json_data = json.loads(json_string)
        TS_segments.put(json_data)

if __name__=="__main__":
    load_weights_params = {
        "path":"/app/data/models/04_20_14_14/",
        "epoch":"final",
    }
    model=model_training.get_model(load_weights_from=load_weights_params,masking_enabled=True,batch_size=1)
    print(model.summary())

    bc = BertClient(ip=BERT_IP)
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()

    # create a sentecen buffer
    sentence_buffer = np.zeros((1,BUFFER_LEN,1024))
    word_buffer     = np.zeros((1,BUFFER_LEN,90,300))
    text_buffer     = ["" for _i in range(BUFFER_LEN)]
    segs            = []

    print("Ready to begin")
    while True:
        # data should be a list of strings, each string is a sentence 
        data = TS_segments.get()
        text_buffer = (text_buffer+data)[-10:]
        word_data = [sentence.split() for sentence in data]
        # everything needs to be encoded now
        bert_data     = data_pre_processing.get_bert_encoding(data,bc)
        word2vec_data = data_pre_processing.get_word2vec_encoding(word_data,max_sentence_len=90,max_sentences=100)
        # fit it into a numpy matrix to fit into the model
        ## add a batch dim
        bert_data = np.expand_dims(bert_data,0)
        word2vec_data = np.expand_dims(word2vec_data,0)
        ## append to buffer window
        sentence_buffer = np.concatenate([sentence_buffer,bert_data],axis=1)
        sentence_buffer = sentence_buffer[:,-10:,:]
        word_buffer     = np.concatenate([word_buffer,word2vec_data],axis=1)
        word_buffer     = word_buffer[:,-10:,:,:]

        # put it into the model
        print("churning the model")
        logits = model({"WE":word_buffer,"SE":sentence_buffer}, training=False).numpy()
        segment_indications = logits>SEG_THRESH
        print("made some cheese")
        # print("logits:\b{}".format(logits.reshape(BUFFER_LEN)))
        # print("segment_indications:\b{}".format(segment_indications.reshape(BUFFER_LEN)))

        # json could not serialise using this method
        # segment_indicators = tf.cast(segment_indications,tf.float32)
        # segment_labels = tf.math.cumsum(segment_indicators).numpy().reshape(BUFFER_LEN)
        # print("segment_labels:\n{}\n".format(segment_labels))
        # segs_string = json.dumps(list(segment_labels))
        # print("segs_string:\n{}\n\n".format(segs_string))

        segment_labels = np.cumsum(segment_indications.reshape(10)*1)
        segment_labels_python = [int(x) for x in segment_labels]
        seg_lab_str    = json.dumps(segment_labels_python)
        print("seg_lab_str:\n{}".format(seg_lab_str))
        client.publish(OUTPUT_TOPIC, seg_lab_str)