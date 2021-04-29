import json,os,re,time
import paho.mqtt.client as mqtt
import threading, queue
import pandas as pd
from string_comparisons import get_WER
from topic_comparisons import get_Pk_error
import numpy as np
from time import gmtime, strftime

CONFIG_PATH = "/app/testconfig.json"

RESULTS_PATH = "/app/results"

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
        data = json.loads(topic_string)
        TC_comparisons.put(data)
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

    # create database to store the results
    results = pd.DataFrame(data={
        "sample_id":[],
        "wer":[],
        "pk":[],
        "command_result":[], # 0 indicates no returned result, 1 indicates incorrect command infered, 2 indicates correct commadn
        })

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
        print("working on index: {}".format(index))
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
        temp  = re.findall(control_tag_command,gt_transcript)
        if len(temp) < 1:
            print("weird issue with input, skipping")
            continue
        command_temp = temp[0]
        command  = re.sub(control_tag_all,"",command_temp)

        print("raw_text len: {}".format(len(raw_text.split())))
        if len(raw_text.split())>200:
            print("raw txt idicates this is a long sample, skipping")
            continue
        ## DS test
        print("Testing DS, this may take a while")
        # message DS to start parsing the audio file:
        client.publish(DS_INPUT_TOPIC, 
            json.dumps({
                "audio_fn":audio_fn,
                "segs":case_data["total_segs"]
                }))
        # print("a")
        # wait 60 seconds for DS to listen and transcribe the data
        # time.sleep(60)
        # wait for DS to respond with text
        # ds_transcript = []
        # while not DS_transcripts.empty():
        ds_transcript = DS_transcripts.get(block=True)
        for element in ds_transcript:
            if len(element)>300:
                print("skipping sample as its huge")
        # print("b")
        # process it
        # exit()
        try:
            wer_out = get_WER("".join(raw_text),"".join(ds_transcript))
        except:
            print("WER issue, skipping sample")
            continue
        # print("c")
        wer = wer_out["WER"]
        print("wer: {}".format(wer))
        ## TS test
        print("Testing TS")
        # pass on text to topic segmenter
        client.publish(TS_INPUT_TOPIC, json.dumps(ds_transcript))
        # print("message sent")
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

        print("command segs:")
        condensed_topics = []
        for sentence in topics:
            words = sentence.split()
            trunc_len = min(len(words),7)
            cond_sent = words[:trunc_len]
            print("\t{}".format(cond_sent))
            # condensed_topics.append(cond_sent)
        print("\n")
        client.publish(TC_INPUT_TOPIC, json.dumps(topics))
        
        # generate template command
        # object ,location ,text 
        command_reference = ""
        if case_data["action"] in ["activate", "deactivate", "increase", "decrease"] :
            command_reference = case_data["action"]+" the "+case_data["object"]
        else:
            command_reference = case_data["action"]+" to "+case_data["object"]

        comparisons = TC_comparisons.get()

        print("command reference:\n\t{}".format(command_reference))
        command_result = {}
        for com_thresh in np.arange(start=0.7,stop=1,step=0.010):
            command_result_local = []
            for command_data in comparisons:
                # conv to df
                df = pd.DataFrame(data=command_data)
                # get those above thresh
                above_thresh = df[df["match"]>com_thresh].sort_values("match",ascending=False)

                if len(above_thresh) == 0: # no command met the thresh
                    command_result_local.append(0)
                else:
                    if above_thresh["command"].iloc[0] ==command_reference:
                        command_result_local.append(2)
                    else:
                        command_result_local.append(1)
            # get best result
            command_result["thresh_{:0.3f}".format(com_thresh)] = max(command_result_local)



        # log this
        sample_data = {
            "sample_id":        case_data["sample_index"],
            "pk":               Pk,
            # "command_result":   command_result, # 0 indicates no returned result, 1 indicates incorrect command infered, 2 indicates correct commadn
        }
        sample_data.update(wer_out)
        sample_data.update(command_result)
        print("sample_data:\n{}\n\n".format(sample_data))
        results = results.append(sample_data,ignore_index=True)

    # after all samples tested, record results
    output_fn = os.path.join(RESULTS_PATH,strftime("%Y_%m_%d-%H_%M_%S.csv", gmtime()))
    with open(output_fn,"w") as fp:
        results.to_csv(fp)