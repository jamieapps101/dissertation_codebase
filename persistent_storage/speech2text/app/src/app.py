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


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("$SYS/#")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Speech Application')
    parser.add_argument('Mode',   metavar='M', nargs='1')
    parser.add_argument('--File',   metavar='F', nargs='1')
    parser.add_argument('--Device', metavar='D', nargs='1')
    args = parser.parse_args()

    # connect to mqtt server
    client = mqtt.Client(
        client_id="", 
        clean_session=True, 
        userdata=None, 
        transport="tcp")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("mqtt.eclipse.org", 1883, 60)
    # start another thread to react to incoming messages
    client.loop_start()

    # load in DS model
    print("loading model")
    model_path = os.path.join(os.getcwd(),"models")
    model = Model(model_path)

    # Start audio with VAD
    # using default params here
    vad_audio = VAD_code.VADAudio(
        aggressiveness=3,# [0,3], 3 filters all non-speech
        device=args.device,
        input_rate=16000, # mic sample rate
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