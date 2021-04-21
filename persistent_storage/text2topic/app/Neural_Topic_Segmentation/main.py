import model_generation
import model_inference
import model_training
import data_pre_processing

from bert_serving.client import BertClient
import json,os,re,time
import paho.mqtt.client as mqtt
import threading, queue
import pandas as pd

CONFIG_PATH = "/app/testconfig.json"

DATA_PATH   = "/app/data/processed"

MQTT_HOST   = "pi4-a"
MQTT_PORT   = "30104"

INPUT_TOPIC = "TS_input"
OUTPUT_TOPIC = "TS_output"



# setup queues to enable callbacks to inform main thread
TS_segments    = queue.Queue()

# setup MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("$SYS/#")
    client.subscript(INPUT_TOPIC)

def on_message(client, userdata, msg):
    global TS_segments
    print(msg.topic+" "+str(msg.payload))
    if   msg.topic == INPUT_TOPIC:
        byte_string = msg.payload
        json_string = byte_string.decode("utf-8")
        json = json.loads(json_string)
        TS_segments.put(json)

if __name__=="__main__":
    load_weights_params = {
        "path":"",
        "epoch":10,
    }
    model=model_training.get_model(load_weights_from=load_weights_params,masking_enabled=True,batch_size=1)
    print(model.summary())

    bc = BertClient(ip="127.0.0.1")

    while True:
        # data should be a list of strings, each string is a sentence 
        data = TS_segments.get()
        word_data = [sentence.split() for sentence in data]
        # everything needs to be encoded now
        bert_data     = data_pre_processing.get_bert_encoding(data,bc)
        word2vec_Data = data_pre_processing.get_word2vec_encoding(word_data,max_sentence_len=90,max_sentences=100)
        # fit it into a numpy matrix to fit into the model