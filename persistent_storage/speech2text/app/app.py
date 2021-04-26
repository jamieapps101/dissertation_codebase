#! /usr/bin/python3 -i

import os
import numpy as np
import glob
from deepspeech import Model
import wavSplit
import webrtcvad
import time
import argparse
import VAD_code
import paho.mqtt.client as mqtt
import signal
import sys
import pyaudio
import threading, queue
import wave 


TEST_MODE = False
TEST_DATA_DIR = "/app"
TEST_CONTROL_TOPIC = "ds_control"
TEST_OUTPUT_TOPIC = "ds_out"
REALTIME_INPUT_DEVICE = 10


def get_env_val(env_name,val):
    if os.environ.get(env_name) is not None:
        return os.environ.get(env_name)
    else:
        return val

TEST_DATA_DIR         = get_env_val("TEST_DATA_DIR",         "/app")
TEST_CONTROL_TOPIC    = get_env_val("TEST_CONTROL_TOPIC",    "ds_control")
TEST_OUTPUT_TOPIC     = get_env_val("TEST_OUTPUT_TOPIC",     "ds_out")
REALTIME_INPUT_DEVICE = get_env_val("REALTIME_INPUT_DEVICE", 10)

REALTIME_INPUT_DEVICE = int(REALTIME_INPUT_DEVICE)

if os.environ.get("TEST_MODE") is not None and os.environ.get("TEST_MODE")=="1":
    TEST_MODE = True

# get a global variable for the callbacks
model = None
data_queue = queue.Queue()

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("$SYS/#")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global data_queue
    print(msg.topic+" "+str(msg.payload))
    if msg.topic == TEST_CONTROL_TOPIC:
        bytes_string = msg.payload
        string_string = bytes_string.decode("utf-8")
        data_queue.put(string_string, block=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Speech Application')
    parser.add_argument('Mode',     metavar='M', nargs=1)
    parser.add_argument('--File',   metavar='F', nargs=1)
    parser.add_argument('--Device', metavar='D', nargs=1,type=int)
    args = parser.parse_args()

    device_index = None
    if args.Device is None:
        # have a look for the camera
        a = pyaudio.PyAudio()
        devs = [a.get_device_info_by_index(i) for i in range(a.get_device_count())]
        for i,dev in enumerate(devs):
            if "HD Pro Webcam" in dev["name"]:
                device_index = i
                print("connecting to:\n{}\n".format(dev))
                break
    else:  
        device_index = args.Device[0]


    # load in DS model
    model_path = os.path.join(os.getcwd(),"models")
    pb = glob.glob(model_path + "/*.pbmm")[0]
    scorer = glob.glob(model_path + "/*.scorer")[0]
    # load them in
    ds = Model(pb)
    ds.enableExternalScorer(scorer)
    model = ds

    if not TEST_MODE:
        # connect to mqtt server
        client = mqtt.Client(
            client_id="", 
            clean_session=True, 
            userdata=None, 
            transport="tcp")

        client.on_connect = on_connect
        client.connect("pi4-a", 30104, 60)
        # start another thread to react to incoming messages
        client.loop_start()
        # Start audio with VAD
        # using default params here
        vad_audio = VAD_code.VADAudio(
            aggressiveness=0,# [0,3], 3 filters all non-speech
            # device=None,
            device=device_index,
            input_rate=32000, # mic sample rate
            file=None)

        print("Listening (ctrl-C to exit)...")
        # get iterator to collect frames of audio data
        frames = vad_audio.vad_collector()

        # setup SIGINT handler
        def signal_handler(sig, frame):
            global client
            client.loop_stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        spinner = None
        stream_context = model.createStream()
        wav_data = bytearray()
        # for each frame
        for frame in frames:
            if frame is not None:
                if spinner: spinner.start()
                # logging.debug("streaming frame")
                stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            else:
                if spinner: spinner.stop()

                # at the end of the activity, get the text and send it to the next node
                text = stream_context.finishStream()
                print("Recognized: {}".format(text))
                client.publish("speech2text/data", text)
                stream_context = model.createStream()

        # end of prog, disconnect mqtt client
        client.loop_stop()
    else:
        # if in test mode
        # connect to mqtt server
        client = mqtt.Client(
            client_id="", 
            clean_session=True, 
            userdata=None, 
            transport="tcp")

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect("pi4-a", 30104, 60)
        # start another thread to react to incoming messages
        client.loop_start()

        while True: 
            file_name = data_queue.get()
            file_path = os.path.join(TEST_DATA_DIR,file_name)
            if os.path.exists(file_path):
                print("transcribing:\n{}".format(file_path))
            else:
                print("could not find:\n{}".format(file_path))
                continue
            fin = wave.open(args.audio, 'rb')
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
            fin.close()

            inference = ds.sttWithMetadata(audio, 1).transcripts[0]
            transcript =  ''.join(token.text for token in inference.tokens)
            client.publish(TEST_OUTPUT_TOPIC,transcript)
            print("I got:\n{}".format(transcript))














# This is just a simplified mockup of the deepspeech demo code, 
# I stole wavSplit file also from the demo code

if __name__ == "__main__" and False:
    # locate model files
    model_path = os.path.join(os.getcwd(),"models")
    print("model_path: {}".format(model_path))
    pb = glob.glob(model_path + "/*.pbmm")[0]
    scorer = glob.glob(model_path + "/*.scorer")[0]
    print("Found Model: {}".format(pb))
    print("Found scorer: {}".format(scorer))
    # load them in
    print("Loading models")
    ds = Model(pb)
    ds.enableExternalScorer(scorer)
    # ds === deep speech model

    # load and pre-process audio
    audio_file = os.path.join(os.getcwd(),"data/testing/audio/2830-3980-0043.wav")
    # audio_file = os.path.join(os.getcwd(),"data/testing/audio/my_name_is_jamie.wav")
    # audio_file = os.path.join(os.getcwd(),"data/testing/audio/hello_liv.wav")
    aggressiveness = 0

    print("Reading and processing: {}".format(audio_file))

    audio, sample_rate, audio_length = wavSplit.read_wave(audio_file)
    assert sample_rate == 16000, "Only 16000Hz input WAV files are supported for now!"
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = wavSplit.frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = wavSplit.vad_collector(sample_rate, 30, 300, vad, frames)

    # we now have the data in the following segments
    # segments, sample_rate, audio_length
    print("we have {} frames".format(len(frames)))
    start = time.time()
    for i, segment in enumerate(segments):
            # Run deepspeech on the chunk that just completed VAD
            print("Processing chunk %002d" % (i,))
            audio = np.frombuffer(segment, dtype=np.int16)
            # Run Deepspeech
            print('Running inference...')
            output = ds.stt(audio)
            print("Transcript: %s" % output)
    end = time.time()
    print("that took: {}".format(end-start))