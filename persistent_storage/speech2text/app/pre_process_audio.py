#! /usr/bin/python3 -i

import json
import random
import os
import pydub
from pydub import AudioSegment,silence
from xml.dom import minidom
import pandas as pd
import numpy as np

def get_child_tags(input_node):
    ret_list = []
    for a in input_node.childNodes:
        if a.nodeType == a.ELEMENT_NODE: 
            ret_list.append(a.tagName) 
        else: 
            ret_list.append("non-element-node") 
    return ret_list

def get_words(sentence_node,sentence_index,paragraph_index,article_title):
    word_nodes = sentence_node.getElementsByTagName("t")
    words = []
    for node_index,word_node in enumerate(word_nodes):
        word_text = word_node.childNodes[0].data
        word_start = -1
        word_end = -1
        if len(word_node.childNodes)>1 and \
            all([prop in word_node.childNodes[1].attributes.keys() for prop in ["start","end"]]):
            word_start = int(word_node.childNodes[1].attributes["start"].value)
            word_end   = int(word_node.childNodes[1].attributes["end"].value)
        words.append({
            "article_title":   article_title,
            "paragraph_index": paragraph_index,
            "sentence_index":  sentence_index,
            "word_index":      node_index,
            "word_text":       word_text,
            "word_start":      word_start,
            "word_end":        word_end,
            "is_command":      False
        })
    return words

def get_sentences(paragraph_node,paragraph_index,article_title):
    all_words = []
    sentence_nodes = paragraph_node.getElementsByTagName("s")
    for sen_index,sen_node in enumerate(sentence_nodes):
        all_words += get_words(sen_node,sen_index,paragraph_index,article_title)
    return all_words

def get_title(xml_data):
    for props_node in xml_data.childNodes[0].childNodes[0].childNodes:
        if props_node.attributes["key"].value == "DC.title":
            return props_node.attributes["value"].value

def get_intro_sections(xml_data):
    article_data = xml_data.getElementsByTagName("article")[0].getElementsByTagName("d")[0]
    # article_meta = xml_data.getElementsByTagName("article")[0].getElementsByTagName("meta")[0]
    article_title = get_title(xml_data)
    article_child_nodes = get_child_tags(article_data)
    # article_sections =  article_data.getElementsByTagName("section")
    article_init_paragraphs =  []
    # print(article_data.childNodes)
    for i in range(len(article_child_nodes)):
        # print("i={}".format(i))
        if article_child_nodes[i]=="p":
            # print("\tarticle_child_nodes[i]={}".format(article_child_nodes[i]))
            # print("\tarticle_data.childNodes[i]={}".format(article_data.childNodes[i]))
            article_init_paragraphs.append(article_data.childNodes[i])

    # print("article_init_paragraphs=\n{}".format(article_init_paragraphs))
    all_words = []
    for paragraph_index,paragraph_node in enumerate(article_init_paragraphs):
        all_words += get_sentences(paragraph_node,paragraph_index,article_title)
    return all_words
    # return None


CONFIG_PATH    = "/app/data/preprocessing_config.json"
SEGMENT_DATA_PATH      = "/app/data/unprocessed/spoken_wikipedia/english"
# SILENCE_THRESH = -16 # in dBFS
SILENCE_THRESH = -38 # in dBFS
PAUSE_THRESH   = 500 # ms

COMMAND_DATA_PATH = "/app/data/unprocessed/fluent_speech_commands_dataset/"
SUMMARY_STATS_PATH = os.path.join(COMMAND_DATA_PATH,"data")
WAV_PATH = COMMAND_DATA_PATH

PROCESSED_DATA_PATH =  "/app/data/processed/"

def get_command_segment_paths(configs):
    paths = pd.read_csv(os.path.join(SUMMARY_STATS_PATH,"train_data.csv"))
    command_mask = None
    for command_spec in configs:
        temp = np.logical_and(
            np.logical_and(
                paths["action"]==command_spec["action"],
                paths["object"]==command_spec["object"]),
            paths["location"]==command_spec["location"])
        if command_mask is None:
            command_mask = temp
        else:
            command_mask = np.logical_or(command_mask,temp)
    return paths.loc[command_mask]



if __name__=="__main__":
    # read in desired audio data files from config json
    config = None
    with open(CONFIG_PATH, "r") as fp:
        config = json.load(fp)
    
    if config is None:
        print("could not read in json file")
        exit(1)

    # load in paths to segments
    command_data = get_command_segment_paths(config["command_segments"])
    available_commands = command_data["transcription"].unique()
    total_commands = command_data.shape[0]

    data_structure = pd.DataFrame(data={
        "sample_index": [],
        "subject":      [],
        "action":       [],
        "object":       [],
        "location":     [],
        "text":         []
    })
    
    subj_index = 0
    max_subj = len(config["wiki_segments"])
    total_samples = 0
    config["wiki_segments_dir"] = os.listdir("/app/data/unprocessed/spoken_wikipedia/english")
    total_subjects = len(config["wiki_segments_dir"])
    # for each audio clip
    while True:
        subj_name = config["wiki_segments_dir"][np.random.randint(total_subjects)]
        print("Generating sample {}/{}, using subj of {}".format(total_samples,config["total_samples"],subj_name))
        # load in audio track
        print("\tloading audio... ")
        subj_path = os.path.join(SEGMENT_DATA_PATH,subj_name)
        try:
            subj_path_contents = os.listdir(subj_path)
        except NotADirectoryError:
            print("just tried to process the README file, that was a little silly...")
            continue
        if "audio.ogg" in subj_path_contents:
            ogg_path = os.path.join(subj_path,"audio.ogg")
        elif "audio1.ogg" in subj_path_contents:
            ogg_path = os.path.join(subj_path,"audio1.ogg")
        else:
            print("No audio file found, skipping")
            continue
        
        segment_audio = None
        try:
            segment_audio = AudioSegment.from_file(ogg_path)
        except pydub.exceptions.CouldntDecodeError :
            print("pydub decode error, skipping sample")
            # subj_index+=1
            continue
        print("\t\t\tdone")


        # load in transcript
        print("\tprocessing transcipt... ")
        xml_path  = os.path.join(subj_path,"aligned.swc") 
        xml_data = None
        if not os.path.exists(xml_path):
            print("could not find\n{}\nskipping sample".format(xml_path))
            continue
        with open(xml_path, "r") as fp:
            xml_data = minidom.parse(fp)
        words = get_intro_sections(xml_data)
        df = pd.DataFrame.from_records(words)
        print("\t\t\tdone")

        # trim segment audio to get only summary section
        try:
            summary_start_time = df[df["word_end"]!=-1]["word_end"].iloc[0]  - 3000 # add 3 seconds onto each end, as there is some error in this 
            summary_end_time   = df[df["word_end"]!=-1]["word_end"].iloc[-1] + 3000
            segment_audio = segment_audio[summary_start_time:summary_end_time]
        except IndexError:
            print("{} gave issues with getting summary, skipping".format(subj_name))
            continue 
        except KeyError:
            print("{} gave issues with getting summary, skipping".format(subj_name))
            continue 

            

        # get all quiet points
        print("\tscanning for quiet points... ")
        q_points = silence.detect_silence(segment_audio,
                min_silence_len=PAUSE_THRESH,
                silence_thresh=SILENCE_THRESH,
                seek_step = 5) # size of step to check for silence, in ms
        
        if len(q_points) <2:
            print("\n\terror, could not find silent point in {} section, skipping".format(subj_name))
            # subj_index+=1
            continue
        print("\t\t\tdone")

        print("\t I found {} quiet points".format(len(q_points)))
        print("\tPicking command, and inserting... ")
        # pick a random one
        rand_insert_index = np.random.randint(len(q_points)-1)
        # pick a random command sample
        rand_command_index = np.random.randint(total_commands)
        command = command_data.iloc[rand_command_index]
        file_path = os.path.join(WAV_PATH,command["path"])
        command_audio = AudioSegment.from_file(file_path)

        # seg_index = 0
        # for q_point_index in range(len(q_points)):
            # seg = segment_audio[q_points[q_point_index][0]:q_points[q_point_index][1]]
        # insert the command sample
        seg_beginning = segment_audio[:q_points[rand_insert_index][1]]
        seg_end       = segment_audio[q_points[rand_insert_index+1][0]:]
        command_seg   = AudioSegment.silent(duration=1000)+\
                        command_audio+\
                        AudioSegment.silent(duration=1000)
        new_segs = [seg_beginning,command_seg,seg_end]
        # write out audio data
        for i,seg in enumerate(new_segs):
            seg.export(os.path.join(PROCESSED_DATA_PATH,"sample_{}_{}.wav".format(total_samples,i)), format="wav")
        print("\t\t\tdone")


        print("\tinserted command at {}ms".format(q_points[rand_insert_index][1]))
        # write out ground true text about sample
        
        print("\tAltering transcript... ")
        #  split segment text in half
        seg_1 =    df[ df["word_end"]   < q_points[rand_insert_index][1]   ]
        seg_2 =    df[ df["word_start"] > q_points[rand_insert_index+1][0] ]
        # now insert the words:
        command_string = command["transcription"]
        print("inserted command string: ",end="")
        for word in command_string.split():
            print(word+" ",end="")
            new_row = {
                    "article_title":   0,
                    "paragraph_index": 0,
                    "sentence_index":  0,
                    "word_index":      0,
                    "word_text":       word, 
                    "word_start":      0,
                    "word_end":        0,
                    "is_command":      True
                }
            seg_1 = seg_1.append(new_row,ignore_index=True)
        print("")
        in_command = False
        # and recombine segments
        text_data = seg_1.append(seg_2,ignore_index=True)
        transcript_path = os.path.join(PROCESSED_DATA_PATH,"sample_{}.txt".format(total_samples))
        with open(transcript_path, "w") as fp:
            for row in text_data.iterrows():
                word = row[1]["word_text"]
                if row[1]["is_command"] == True and in_command==False:
                    in_command=True
                    fp.write("<command>")
                filtered_word = ''.join(filter(str.isalpha, word))
                if len(filtered_word)>0:
                    fp.write(filtered_word + " ")
                if row[1]["is_command"] == False and in_command==True:
                    in_command=False
                    fp.write("</command>")

        print("\t\t\tdone")

        data_structure = data_structure.append({
                "sample_index": total_samples,
                "subject":      subj_name,
                "action":       command["action"],
                "object":       command["object"],
                "location":     command["location"],
                "text":         command["transcription"],
                "total_segs":   3
            },ignore_index=True)

        print("\tfinished sample {}".format(subj_index))

        # subj_index+=1
        total_samples+=1
        if total_samples == config["total_samples"]:
            break
        # if subj_index>=len(config["wiki_segments"]):
            # subj_index -= len(config["wiki_segments"])
            
        print("\n")
    print("completed sampling")
    summary_path = os.path.join(PROCESSED_DATA_PATH,"summary.csv")
    data_structure.to_csv(summary_path)