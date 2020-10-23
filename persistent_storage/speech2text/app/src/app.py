import os
import numpy as np
import glob
from deepspeech import Model
import wavSplit
import webrtcvad
import time

# This is just a simplified mockup of the deepspeech demo code, 
# I stole wavSplit file also from the demo code

if __name__ == "__main__":
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