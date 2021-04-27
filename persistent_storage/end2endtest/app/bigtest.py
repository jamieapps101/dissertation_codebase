import json,os,re,time
import paho.mqtt.client as mqtt
import threading, queue
import pandas as pd
from string_comparisons import get_WER
from topic_comparisons import get_Pk_error
import numpy as np

CONFIG_PATH = "/app/testconfig.json"

DATA_PATH   = "/speech2text/app/data/processed/"

MQTT_HOST   = "127.0.0.1"
MQTT_PORT   = 1883

DS_INPUT_TOPIC = "ds_control"
TS_INPUT_TOPIC = "TS_input"
TC_INPUT_TOPIC = "TC_input"

DS_OUTPUT_TOPIC = "ds_out"
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
    # client.subscribe("$SYS/#")
    client.subscribe(DS_OUTPUT_TOPIC)
    client.subscribe(TS_OUTPUT_TOPIC)
    client.subscribe(TC_OUTPUT_TOPIC)

def on_message(client, userdata, msg):
    global DS_transcripts
    global TS_segments
    global TC_comparisons
    # print(msg.topic+" "+str(msg.payload))
    if   msg.topic == DS_OUTPUT_TOPIC:
        print("Received DS output")
        transcript = msg.payload.decode("utf-8")
        # print("transcript:\n\n{}\n\n".format(transcript))
        DS_transcripts.put(json.loads(transcript))
    elif msg.topic == TS_OUTPUT_TOPIC:
        print("Received TS output")
        topic_string = msg.payload.decode("utf-8")
        topics = json.loads(topic_string)
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
    # test_data_ref = None
    # with open(os.path.join(DATA_PATH, "summary.csv"),"r") as fp:
        # test_data_ref = pd.DataFrame.from_csv(fp)
    # if test_data_ref is None:
        # print("could not load test data summary")
        # exit(1)

    #  setup MQTT connection
    print("setup MQTT connection")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()
    time.sleep(3)
    print("done")

    data_summary_path = "/speech2text/app/data/processed/summary.csv"

    data_summary = pd.read_csv(data_summary_path)    
    # exit()
    # begin looping through test cases
    for index,case_data in data_summary.iterrows():

        if case_data["sample_index"]==0:
            continue
        # generate path information
        audio_fn = "sample_{}".format(int(case_data["sample_index"]))
        # audio_path      = os.path.join(DATA_PATH,audio_fn)
        # if not os.path.exists(audio_path):
            # print("could not locate:{}\n skipping test case".format(audio_path))
            # continue
        transcript_fn = "sample_{}.txt".format(int(case_data["sample_index"]))
        transcript_path = os.path.join(DATA_PATH,transcript_fn)
        if not os.path.exists(transcript_path):
            print("could not locate:{}\n skipping test case".format(transcript_path))
            continue

        # extract data from transcript:
        gt_transcript = None
        with open(transcript_path,"r") as fp:
            gt_transcript = fp.read()
        control_tag_all      = re.compile(r"\<\/?command\>")
        control_tag_command  = re.compile(r"\<command\>[\w\s]+\<\/command\>")
        raw_text = re.sub(control_tag_all,"",gt_transcript)
        topics   = re.split(control_tag_all,gt_transcript)
        command_temp  = re.findall(control_tag_command,gt_transcript)[0]
        command  = re.sub(control_tag_all,"",command_temp)


        ## DS test
        print("Testing DS, this may take a while")
        # message DS to start parsing the audio file:
        client.publish(DS_INPUT_TOPIC, 
            json.dumps({
                "audio_fn":audio_fn,
                "segs":case_data["total_segs"]
                }))
        # wait 60 seconds for DS to listen and transcribe the data
        # time.sleep(60)
        # wait for DS to respond with text
        # ds_transcript = []
        # while not DS_transcripts.empty():
        ds_transcript = DS_transcripts.get(block=True)
        # process it
        wer = get_WER("".join(gt_transcript),"".join(ds_transcript))
        print("wer: {}".format(wer))
        ## TS test
        print("Testing TS")
        # pass on text to topic segmenter
        client.publish(TS_INPUT_TOPIC, json.dumps(ds_transcript))
        # wait for segments to be reterned
        topics_window = TS_segments.get()
        topics = topics_window[-len(ds_transcript):]
        topics_hype = [t-topics[0] for t in topics]
        topics_gt = list(range(len(ds_transcript)))
        # exit()
        Pk = get_Pk_error(topics_hype,topics_gt,1)
        print("Pk:{}".format(Pk))

        
        
        # TC test
        print("Testing TC")
        # join together all segments deemed to be the same topic
        unique_seg_IDs = np.unique(np.array(topics_hype))
        topics = []
        topics_hype_np = np.array(topics_hype)
        for topic_label in unique_seg_IDs:
            seg_indexes = np.where(topics_hype_np==topic_label)[0]
            temp = ""
            for index in seg_indexes:
                temp += ds_transcript[index]
            topics.append(temp)

        client.publish(TC_INPUT_TOPIC, json.dumps(topics))
        command = TC_comparisons.get()
        # log this