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
import shlex
import subprocess

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

TEST_MODE = True
TEST_DATA_DIR = "/app/data/processed"
TEST_CONTROL_TOPIC = "ds_control"
TEST_OUTPUT_TOPIC = "ds_out"
REALTIME_INPUT_DEVICE = 10


def get_env_val(env_name,val):
    if os.environ.get(env_name) is not None:
        return os.environ.get(env_name)
    else:
        return val

TEST_DATA_DIR         = get_env_val("TEST_DATA_DIR",         TEST_DATA_DIR)
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
    client.subscribe(TEST_CONTROL_TOPIC)
    print("should be subbed")

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global data_queue
    print(msg.topic+" "+str(msg.payload))
    if msg.topic == TEST_CONTROL_TOPIC:
        bytes_string = msg.payload
        string_string = bytes_string.decode("utf-8")
        data_queue.put(string_string, block=False)

# from deep speech
def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)


if __name__ == "__main__":
    print("TEST_MODE: {}".format(TEST_MODE))

    # parser = argparse.ArgumentParser(description='Deep Speech Application')
    # parser.add_argument('Mode',     metavar='M', nargs=1)
    # parser.add_argument('--File',   metavar='F', nargs=1)
    # parser.add_argument('--Device', metavar='D', nargs=1,type=int)
    # args = parser.parse_args()

    device_index = None
    if False:
        # have a look for the camera
        a = pyaudio.PyAudio()
        devs = [a.get_device_info_by_index(i) for i in range(a.get_device_count())]
        for i,dev in enumerate(devs):
            if "HD Pro Webcam" in dev["name"]:
                device_index = i
                print("connecting to:\n{}\n".format(dev))
                break


    # load in DS model
    model_path = os.path.join(os.getcwd(),"models")
    pb = glob.glob(model_path + "/*.pbmm")[0]
    scorer = glob.glob(model_path + "/*.scorer")[0]
    # load them in
    ds = Model(pb)
    ds.enableExternalScorer(scorer)
    model = ds
    desired_sample_rate = ds.sampleRate()

    print("TEST_MODE: {}".format(TEST_MODE))

    if not TEST_MODE:
        # connect to mqtt server
        client = mqtt.Client(
            client_id="", 
            clean_session=True, 
            userdata=None, 
            transport="tcp")

        client.on_connect = on_connect
        client.connect("127.0.0.1", 1883, 60)
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
        client.connect("127.0.0.1", 1883, 60)
        # start another thread to react to incoming messages
        client.loop_start()
        print("beginning looping")
        while True: 
            file_name = data_queue.get()
            print("reading:\n{}".format(file_name))
            file_path = os.path.join(TEST_DATA_DIR,file_name)
            if os.path.exists(file_path):
                print("transcribing:\n{}".format(file_path))
            else:
                print("could not find:\n{}".format(file_path))
                continue
            fin = wave.open(file_path, 'rb')
            fs_orig = fin.getframerate()
            if fs_orig != desired_sample_rate:
                print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, desired_sample_rate), file=sys.stderr)
                fs_new, audio = convert_samplerate(file_path, desired_sample_rate)
            else:
                audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
            # audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
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