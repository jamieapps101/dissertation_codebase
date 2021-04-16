#! /usr/bin/python3 -i

import json
import random
import os
from pydub import AudioSegment
from xml.dom import minidom
import pandas as pd

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
DATA_PATH      = "/app/data/spoken_wikipedia/english"
SILENCE_THRESH = -16 # in dBFS
PAUSE_THRESH   = 500 # ms

if __name__=="__main__":
    # read in desired audio data files from config json
    config = None
    with open(CONFIG_PATH, "r") as fp:
        config = json.load(fp)
    
    if config is None:
        print("could not read in json file")
        exit(1)
    
    # for each audio clip
    for file_name in config["files"]:
        file_path = os.path.join(DATA_PATH,file_name)
        # segment_audio = AudioSegment.from_file(file_path)
        # get all quiet points
        # q_points = segment_audio.detect_silence(
            # min_silence_len=PAUSE_THRESH,
            # silence_thresh=SILENCE_THRESH
            # seek_step = 1) # size of step to check for silence, in ms
        # pick a random one
        # pick a random command sample
        # insert the command sample
        # write out audio data
        # write out ground true text about sample

    subj_path = os.path.join(DATA_PATH,"YouTube")
    xml_path  = os.path.join(subj_path,"aligned.swc") 
    xml_data = None
    with open(xml_path, "r") as fp:
        xml_data = minidom.parse(fp)

    words = get_intro_sections(xml_data)

    df = pd.DataFrame.from_records(words)
