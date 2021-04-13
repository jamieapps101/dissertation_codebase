#! /usr/bin/python3 -i

# this app reads in a list of existing commands from a file, and encodes
# them in BERT, it then takes incoming text segments, encodes them and 
# compares them to the encoded commands using cosine similarity. if past
# a threshold, then it outputs the index of the commadn to the next section
from bert_serving.client import BertClient
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
from scipy import spatial
import json
import os

def get_env_val(env_name,val):
    if os.environ.get(env_name) is not None:
        return os.environ.get(env_name)
    else:
        return val

JSON_COMMAND_FILE_PATH = get_env_val("JSON_COMMAND_FILE_PATH" ,"/app/config/commands.json")
BERT_IP                = get_env_val("BERT_IP"                ,"10.80.0.1")
# MQTT_IP                = get_env_val("MQTT_IP"                ,"pi4-a")
MQTT_IP                = get_env_val("MQTT_IP"                ,"192.168.0.72")
MQTT_PORT              = get_env_val("MQTT_PORT"              ,"30104")
INCOMING_TEXT_TOPIC    = get_env_val("INCOMING_TEXT_TOPIC"    ,"in_topic" )
OUTGOING_COMM_TOPIC    = get_env_val("OUTGOING_COMM_TOPIC"    ,"out_topic" )
MATCH_THRESH           = get_env_val("MATCH_THRESH"           ,"0.8")

MQTT_PORT              = int(MQTT_PORT)
MATCH_THRESH           = float(MATCH_THRESH)


class CommandDB:
    def __init__(self,commands,encodings,thresh=0.5):
        self.commands  = commands
        self.encodings = encodings
        self.thresh = thresh
    
    def get_matches(self, input_encoding):
        matches = pd.DataFrame({"signal":[],"command":[],"match":[]})
        for command_index in range(self.commands.shape[0]):
            sim = 1-spatial.distance.cosine(self.encodings[command_index],input_encoding)
            db_slice = self.commands.iloc[command_index] 
            if sim >= self.thresh:
                entry = {"signal":db_slice["signal"],"command":db_slice["command"],"match":sim}
                matches = matches.append(entry,ignore_index=True)
        return matches.sort_values(["match"],ascending=False)

# setup global vars
bert_client_connection = None
command_db             = None

# setup callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(INCOMING_TEXT_TOPIC)

def on_message(client, userdata, msg):
    global bert_client_connection
    global command_db
    if msg.topic == INCOMING_TEXT_TOPIC:
        print(msg.topic+" "+str(msg.payload))
        byte_string = msg.payload 
        string_string = byte_string.decode("utf-8")
        # encode topic
        encode = bert_client_connection.encode([string_string])
        # get cosine sims
        # get those above thresh
        matches = command_db.get_matches(encode)
        if matches.shape[0] < 1:
            print("no matches above thresh")
        else:
            print("best match:\n{}".format(matches.iloc[0]))
            client.publish(OUTGOING_COMM_TOPIC,matches.iloc[0]["signal"])


if __name__ == "__main__":
    # setup connection to bert
    bert_client_connection = BertClient(ip=BERT_IP) 

    # first read in commands, then pre-encode them
    commands_dict = None
    with open(JSON_COMMAND_FILE_PATH,"r") as fp:
        commands_dict = json.load(fp)
    if commands_dict is None:
        print("could not load commands")
        exit(1)
    command_pd = pd.DataFrame.from_dict(commands_dict,orient="index")
    encodings = bert_client_connection.encode(list(command_pd["command"]))
    command_db = CommandDB(command_pd,encodings,MATCH_THRESH)

    # then activate mqtt listener
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_IP, MQTT_PORT, 60)

    client.loop_forever()