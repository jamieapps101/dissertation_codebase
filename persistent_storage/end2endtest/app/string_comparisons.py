# This file contains the functions used to compare 
# the strings give by DS with the ground truth labels
import numpy as np
np.set_printoptions(edgeitems=10,linewidth=100000)
import pandas as pd 
import re

def get_coord(a_pos,b_pos,a_len):
    return b_pos*a_len+a_pos

def get_pos(coord,a_len):
    a_pos = int(coord%a_len)
    b_pos = int(np.floor(coord/a_len))
    return a_pos,b_pos
    

def gen_transition_mat(a,b):
    a_len = len(a)
    b_len = len(b)
    ab    = (a_len+1)*(b_len+1)
    # this is the transition matrix representing that graph
    
    # this produces a grid style matrix
    # start with a matrix with very high weights between all nodes
    transition_mat = np.ones((ab,ab))*np.inf
    for a_index in range(a_len+1):
        for b_index in range(b_len+1):
            base_index     = get_coord(a_index,b_index,  a_len+1)
            if a_index!= a_len:
                downone_index  = get_coord(a_index+1,b_index,a_len+1)
                transition_mat[base_index,downone_index]  = 1
            if b_index!= b_len:
                alongone_index = get_coord(a_index,b_index+1,a_len+1)
                transition_mat[base_index,alongone_index] = 1
            # this produces diagonals with 0 cost connections, for connections
            # where elements are identical
            if a_index!= a_len and b_index!= b_len and a[a_index] == b[b_index]:
                diagonalone_index = get_coord(a_index+1,b_index+1,a_len+1) 
                transition_mat[base_index,diagonalone_index] = 0
    return transition_mat

def perform_dijkstra(start_node,end_node,transmat):
    # first generate a matrix to store the best paths to each node
    #  the index represent the nodes serialised id
    path_data = pd.DataFrame(data={
        "visited"  : np.zeros(transmat.shape[0]),
        "cost"      : np.ones(transmat.shape[0])*np.inf,
        "prev_node" : -np.ones(transmat.shape[0]),
    })
    path_data.at[start_node,"visited"]=1
    path_data.at[start_node,"cost"]=0
    node_queue = [start_node]
    while len(node_queue):
        # sort the node queue
        node_queue.sort()
        # remove the shortest one
        current_node = node_queue[0]
        node_queue = node_queue[1:]
        if current_node==end_node:
            break
        path_data.at[current_node,"visited"] = 1
        connected_nodes = np.argwhere(transmat[current_node,:]<np.inf)
        # not sure why arg where is giving two col output here, but the -1 fixes that
        for node in connected_nodes[:,-1]:
            node_cost = path_data.at[current_node,"cost"]+transmat[current_node,node]
            # if we've not visited that node
            if path_data.at[node,"visited"]==0:
                path_data.at[node,"visited"]=1
                path_data.at[node,"cost"]=node_cost
                path_data.at[node,"prev_node"]=current_node
                node_queue.append(node)
            else:
                if path_data.at[node,"cost"] > node_cost:
                    path_data.at[node,"cost"] = node_cost
                    path_data.at[node,"prev_node"]=current_node
    return path_data

def get_shortest_path(start_node,end_node,transmat):
    path_data = perform_dijkstra(start_node,end_node,transmat)
    #get the shortest path
    path = [end_node]
    prev_node = -1
    while prev_node != start_node:
        prev_node = path_data.at[path[-1],"prev_node"]
        path.append(prev_node)
    path.reverse()
    return path

def get_actions(a,b,path):
    instructions = []
    a_len = len(a)
    for i in range(len(path)-1):
        a_pos_1,b_pos_1 = get_pos(path[i],a_len+1)
        a_pos_2,b_pos_2 = get_pos(path[i+1],a_len+1)
        a_pos_delta = a_pos_2-a_pos_1
        b_pos_delta = b_pos_2-b_pos_1
        if a_pos_delta == 0:
            # no a pos change, therefore there is a b pos change
            # this is an insertion
            instructions.append({"key":"insert","value":b[b_pos_1]})
        elif b_pos_delta == 0:
            # no b pos change, therefore there is an a pos change
            # this is a deletion
            instructions.append({"key":"delete","value":a[a_pos_1]})
        else:
            # both a and b pos have changed, therefore no 
            # insertion or deletion
            instructions.append({"key":"keep","value":b[b_pos_1]})
    return instructions


# This alg works for two lists of numbers
# it works grom from a to b
def myers_alg(a,b):
    # first produce a graph, with m rows where m is the length of the initial sequence
    # and n cols where n is the length of the target sequence
    print("sc - j")
    transition_mat = gen_transition_mat(a,b)

    # now perform dijkstras alg for going from top right pos (0,0) to bottom
    # right graph node (n,m)
    print("sc - k")
    # gen_transition_mat(a,b)
    start_node     = 0
    print("sc - l")
    end_node       = (len(a)+1)*(len(b)+1)-1
    print("sc - m")
    path = get_shortest_path(start_node,end_node,transition_mat)
    print("sc - n")
    actions = get_actions(a,b,path)
    return actions

def encode_words(word_table, words):
    return [np.argwhere(word_table==word)[0,0] for word in words]

def decode_words(word_table, enc_words):
    return [word_table[val] for val in enc_words]

# a and b are strings, with only words, no punctuation (strings are split on
# white space, so "hello"!="hello!")
def myers_alg_string(a,b):
    # convert to unique identifers
    print("sc - d")
    words_raw = a.split()+b.split()
    print("sc - e")
    word_table = pd.unique(words_raw)
    print("sc - f")
    a_encoded = encode_words(word_table,a.split())
    print("sc - g")
    b_encoded = encode_words(word_table,b.split())
    print("sc - h")
    actions = myers_alg(a_encoded,b_encoded)
    print("sc - i")
    return actions,word_table

def pretty_print(a,b):
    actions,word_table = myers_alg_string(a,b)
    for action in actions:
        # print(action)
        if action["key"] == "keep":
            print(decode_words(word_table,[action["value"]])[0])
        if action["key"] == "delete":
            print("-- " + decode_words(word_table,[action["value"]])[0])
        if action["key"] == "insert":
            print("++ " + decode_words(word_table,[action["value"]])[0])

def calc_WER(actions):
    S = 0
    D = 0
    I = 0
    C = 0
    print("sc - b")
    last_action = None
    for action in actions:
        if action["key"]=="keep": # the action must be to keep
            C+=1
            last_action="keep"
        else:
            if last_action!="subst":
                if action["key"]=="insert":
                    if last_action=="delete":
                        D-=1
                        S+=1
                        last_action="subst"
                    else:
                        I+=1
                        last_action="insert"

                elif action["key"]=="delete":
                    if last_action=="insert":
                        I-=1
                        S+=1
                        last_action="subst"
                    else:
                        D+=1
                        last_action="delete"

    print("sc - c")
    WER = (S+D+I)/(S+D+C)
    print("S:{}".format(S))
    print("D:{}".format(D))
    print("I:{}".format(I))
    print("C:{}".format(C))
    out = {
        "S":    S,
        "D":    D,
        "I":    I,
        "C":    C,
        "WER":  WER
    }
    return out


# input strings to this
def get_WER(a,b):
    print("sc - a")
    actions,_word_table = myers_alg_string(a,b)
    print("sc - b")
    return calc_WER(actions)

if __name__=="__main__":
    a = "mary had a little lamb it was very fluffy"
    b = "mark had a mouldy tin of baked beans they were very fluffy"

    a = "we wanted people to know that we’ve got something brand new and essentially this product is uh what we call disruptive changes the way that people interact with technology"
    b = "We wanted people to know that how to me where i know and essentially this product is what we call scripted changes the way people are rapid technology"

    actions,word_table = myers_alg_string(a,b)
    for action in actions:
        # print(action)
        if action["key"] == "keep":
            print(decode_words(word_table,[action["value"]])[0])
        if action["key"] == "delete":
            print("-- " + decode_words(word_table,[action["value"]])[0])
        if action["key"] == "insert":
            print("++ " + decode_words(word_table,[action["value"]])[0])
    WER = calc_WER(actions)
    print("WER={}".format(WER))

    
