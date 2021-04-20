import json,os,re,time
import paho.mqtt.client as mqtt
import threading, queue
import pandas as pd
from string_comparisons import get_WER
from topic_comparisons import get_Pk_error

CONFIG_PATH = "/app/testconfig.json"

DATA_PATH   = "/app/data/processed"

MQTT_HOST   = "pi4-a"
MQTT_PORT   = "30104"

DS_INPUT_TOPIC = "DS_input"
TS_INPUT_TOPIC = "TS_input"
TC_INPUT_TOPIC = "TC_input"

DS_OUTPUT_TOPIC = "DS_output"
TS_OUTPUT_TOPIC = "TS_output"
TC_OUTPUT_TOPIC = "TC_output"

# setup queues to enable callbacks to inform main thread
DS_transcripts = queue.Queue()
TS_segments    = queue.Queue()
TC_comparisons = queue.Queue()

# setup MQTT callbacks
def on_connect(client, userdata, flags, rc):
    global DS_transcripts
    global TS_segments
    global TC_comparisons
    print("Connected with result code "+str(rc))
    client.subscribe("$SYS/#")
    client.subscript(DS_OUTPUT_TOPIC)
    client.subscript(TS_OUTPUT_TOPIC)
    client.subscript(TC_OUTPUT_TOPIC)

def on_message(client, userdata, msg):
    global DS_transcripts
    global TS_segments
    global TC_comparisons
    print(msg.topic+" "+str(msg.payload))
    if   msg.topic == DS_OUTPUT_TOPIC:
        print("Received DS output")
        transcript = msg.payload.decode("utf-8")
        DS_transcripts.put(transcript)
    elif msg.topic == TS_OUTPUT_TOPIC:
        print("Received TF output")
        topic_string = msg.payload.decode("utf-8")
        topics = topic_string.split("|")
        TS_segments.put(topics)
    elif msg.topic == TC_OUTPUT_TOPIC:
        print("Received TC output")
        topic_string = msg.payload.decode("utf-8")
        TC_comparisons.put(topic_string)
    else:
        print("Received non-data message")


if __name__=="__main__":
    # load in text config
    config = None
    with open(CONFIG_PATH,"r") as fp:
        config = json.load(fp)
    if config is None:
        print("could not load config")
        exit(1)
    
    # load in summary data CSV
    test_data_ref = None
    with open(os.path.join(DATA_PATH, "summary.csv"),"r") as fp:
        test_data_ref = pd.DataFrame.from_csv(fp)
    if test_data_ref is None:
        print("could not load test data summary")
        exit(1)

    #  setup MQTT connection
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 60)

    # begin looping through test cases
    for case in config["test_cases"]:
        # generate path information
        audio_path      = os.path.join(DATA_PATH,"{}.mp3".format(case))
        if not os.path.exists(audio_path):
            print("could not locate:{}\n skipping test case".format(audio_path))
            continue
        transcript_path = os.path.join(DATA_PATH,"{}.txt".format(case))
        if not os.path.exists(transcript_path):
            print("could not locate:{}\n skipping test case".format(transcript_path))
            continue

        # extract data from transcript:
        gt_transcript = None
        with open(transcript_path,"r") as fp:
            gt_transcript = fp.read()
        control_tag_all   = re.compile(r"\</?command\>")
        control_tag_command  = re.compile(r"\<command\>\w*\</command\>")
        raw_text = re.sub(control_tag_all,"",gt_transcript)
        topics   = re.split(control_tag_all,gt_transcript)
        command  = re.findall(control_tag_command,gt_transcript)
        command  = re.sub(control_tag_all,"",command)


        ## DS test
        print("Testing DS, this may take a while")
        # message DS to start parsing the audio file:
        client.publish(DS_INPUT_TOPIC, audio_path)
        # wait 60 seconds for DS to listen and transcribe the data
        time.sleep(60)
        # wait for DS to respond with text
        ds_transcript = []
        while not DS_transcripts.empty():
            ds_transcript=DS_transcripts.get(block=False)
        # process it
        wer = get_WER("".join(gt_transcript),"".join(ds_transcript))

        ## TS test
        print("Testing TS")
        # pass on text to topic segmenter
        client.publish(TS_INPUT_TOPIC, ds_transcript)
        # wait for segments to be reterned
        topics = TS_segments.get()
        # process it
        Pk = get_Pk_error([],[],5)

        
        
        # TC test
        print("Testing TC")
        # pass on segments to segment comparison
        # wait for copmmrisons to be reterned
        command = TC_comparisons.get()
        # log this