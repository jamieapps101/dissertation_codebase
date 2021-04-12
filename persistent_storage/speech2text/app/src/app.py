import os
import numpy as np
import glob
from deepspeech import Model
import wavSplit
import webrtcvad
import time
import argparse
import VAD_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Speech Application')
    parser.add_argument('Mode',   metavar='M', nargs='1')
    parser.add_argument('--File',   metavar='F', nargs='1')
    parser.add_argument('--Device', metavar='D', nargs='1')
    args = parser.parse_args()

    # connect to mqtt server
    

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

    spinner = None
    stream_context = model.createStream()
    wav_data = bytearray()
    # for each frame
    for frame in frames:
        if frame is not None:
            if spinner: spinner.start()
            logging.debug("streaming frame")
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            # if ARGS.savewav: wav_data.extend(frame)
        else:
            if spinner: spinner.stop()
            # logging.debug("end utterence")
            # if ARGS.savewav:
                # vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                # wav_data = bytearray()
            text = stream_context.finishStream()
            print("Recognized: %s" % text)
            stream_context = model.createStream()
















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