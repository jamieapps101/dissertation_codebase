from bert_serving.client import BertClient
import json,os,re,time
import paho.mqtt.client as mqtt
import threading, queue
import pandas as pd
import numpy as np

CONFIG_PATH = "/app/testconfig.json"

DATA_PATH   = "/app/data/processed"

MQTT_HOST   = "127.0.0.1"
MQTT_PORT   = 1883

INPUT_TOPIC = "TS_input"
OUTPUT_TOPIC = "TS_output"


# setup MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(OUTPUT_TOPIC)

def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))



if __name__=="__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()