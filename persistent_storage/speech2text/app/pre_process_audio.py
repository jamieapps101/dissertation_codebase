#! /usr/bin/python3 -i

import json
import random
import os
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
            "article_title": article_title,
            "paragraph_index":paragraph_index,
            "sentence_index":sentence_index,
            "word_index":node_index,
            "word_text":word_text,
            "word_start":word_start,
            "word_end":word_end,
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
SILENCE_THRESH = -20 # in dBFS
PAUSE_THRESH   = 500 # ms

COMMAND_DATA_PATH = "/app/data/unprocessed/fluent_speech_commands_dataset/"
SUMMARY_STATS_PATH = os.path.join(COMMAND_DATA_PATH,"data")
WAV_PATH = COMMAND_DATA_PATH

PROCESSED_DATA_PATH =  "/app/data/unprocessed/"

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
    
    subj_index = 0
    max_subj = len(config["wiki_segments"])
    # for each audio clip
    while True:
        subj_name = config["wiki_segments"][subj_index]
        # load in audio track
        print("loading audio... ",end="")
        subj_path = os.path.join(SEGMENT_DATA_PATH,subj_name)
        ogg_path = os.path.join(subj_path,"audio.ogg")
        segment_audio = AudioSegment.from_file(ogg_path)
        print("done")
        # get all quiet points
        print("scanning for quiet points... ",end="")
        q_points = silence.detect_silence(segment_audio,
                min_silence_len=PAUSE_THRESH,
                silence_thresh=SILENCE_THRESH,
                seek_step = 1) # size of step to check for silence, in ms
        
        if len(q_points) == 1:
            print("\nerror, could not find silent point in {} section, skipping".format(subj_name))
            continue
        print("done")

        print("Picking command, and inserting... ",end="")
        # pick a random one
        rand_insert_index = np.random.randint(len(q_points))
        # pick a random command sample
        rand_command_index = np.random.randint(total_commands)
        command = command_data.iloc[rand_command_index]
        file_path = os.path.join(WAV_PATH,command["path"])
        command_audio = AudioSegment.from_file(file_path)
        # insert the command sample
        seg_beginning = segment_audio[:q_points[rand_insert_index][1]]
        seg_end = segment_audio[:q_points[rand_insert_index+1][0]]
        new_seg = seg_beginning+command_audio+seg_end
        # write out audio data
        new_seg.export(os.path.join(PROCESSED_DATA_PATH,"sample_{}.mp3".format(subj_index)), format="mp3")
        print("done")
        
        
        # write out ground true text about sample
        print("processing transcipt... ",end="")
        xml_path  = os.path.join(subj_path,"aligned.swc") 
        xml_data = None
        with open(xml_path, "r") as fp:
            xml_data = minidom.parse(fp)

        words = get_intro_sections(xml_data)

        df = pd.DataFrame.from_records(words)

        print("done")


        if subj_index == config["total_samples"]:
            break
        subj_index+=1
        print("\n")
