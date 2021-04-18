# This file contains the functions used to compare 
# the strings give by DS with the ground truth labels
import numpy as np
import pandas as pd 

def get_coord(a_pos,b_pos,a_len):
    return b_pos*a_len+a_pos

def gen_transition_mat(a,b):
    a_len = len(a)
    b_len = len(b)
    ab    = a_len*b_len
    # this is the transition matrix representing that graph
    
    # this produces a grid style matrix
    # start with a matrix with very high weights between all nodes
    transition_mat = np.ones((ab,ab))*np.inf
    for a_index in range(a_len-1):
        for b_index in range(b_len-1):
            base_index     = get_coord(a_index,b_index,a_len)
            downone_index  = get_coord(a_index+1,b_index,a_len)
            alongone_index = get_coord(a_index,b_index+1,a_len)
            transition_mat[base_index,downone_index]  = 1
            transition_mat[base_index,alongone_index] = 1
            # this produces diagonals with 0 cost connections, for connections
            # where elements are identical
            if a[a_index] == b[b_index]:
                diagonalone_index = get_coord(a_index+1,b_index+1,a_len) 
                transition_mat[base_index,diagonalone_index] = 0
    return transition_mat

def get_shortest_path(start_node,end_node,transmat):
    # a_len = len(a)
    # b_len = len(b)
    # first generate a matrix to store the best paths to each node
    #  the index represent the nodes serialised id
    path_data = pd.DataFrame(data={
        "visited"  : np.zeros(transmat.shape[0]),
        "cost"      : np.ones(transmat.shape[0])*np.inf,
        "prev_node" : -np.ones(transmat.shape[0]),
    })

    # start_node = get_coord(start_y,start_x,a_len)
    # end_node   = get_coord(end_y,  end_x,  a_len)

    path_data.at[start_node,"visited"]=1
    path_data.at[start_node,"cost"]=0

    node_queue = [start_node]
    while len(node_queue):
        # sort the node queue
        node_queue.sort()
        # remove the shortest one
        current_node = node_queue[0]
        # print("type(current_node[\"node_id\"]): {}".format(type(current_node["node_id"])))
        #  this is hacky, but the line above returns a numpy64 int when   
        # node_queue has one item in it, and a list otherwise
        # if not isinstance(current_node["node_id"],list):
            # current_node["node_id"] = [current_node["node_id"]]
        # if en
        node_queue = node_queue[1:]
        print("node_queue: {}".format(node_queue))
        # compare with end state
        print("current_node[\"node_id\"]: {}".format(current_node))
        if current_node==end_node:
            print("reached end node")
            break


        path_data.at[current_node,"visited"] = 1
        connected_nodes = np.argwhere(transmat[current_node,:]<np.inf)
        # print("connected_nodes:\n{}".format(connected_nodes))
        # not sure why arg where is giving two col output here, but the -1 fixes that
        print("connected_nodes[:,-1]: {}".format(connected_nodes[:,-1]))
        for node in connected_nodes[:,-1]:
            print("node:{}".format(node))
            print("current_node[\"cost\"]:{}".format(path_data.at[current_node,"cost"]))
            print("additional cost:{}".format(transmat[current_node,node]))

            node_cost = path_data.at[current_node,"cost"]+transmat[current_node,node]

            print("node_cost: {}".format(node_cost))
            # if we've not visited that node
            if path_data.at[node,"visited"]==0:
                path_data.at[node,"visited"]=1
                path_data.at[node,"cost"]=node_cost

                path_data.at[node,"prev_node"]=current_node
                print("node_queue: {}".format(node_queue))
                node_queue.append(node)
                print("adding node {} to queue with cost {}".format(node,node_cost))
            else:
                print("path_data.at[node,\"cost\"]: {}".format(path_data.at[node,"cost"]))
                if path_data.at[node,"cost"] > node_cost:
                    path_data.at[node,"cost"] = node_cost
                    # path_data.at[node,"prev_node"]=current_node["node_id"][0]
                    path_data.at[node,"prev_node"]=current_node

                    print("updating node {}'s cost to {}".format(node,node_cost))
                else:
                    print("not updating node {}'s cost".format(node))
            print("")
    return path_data



# This alg works for two lists of numbers
# it works grom from a to b
def myers_alg(a,b):
    # first produce a graph, with m rows where m is the length of the initial sequence
    # and n cols where n is the length of the target sequence
    transition_mat = gen_transition_mat(a,b)

    # now perform dijkstras alg for going from top right pos (0,0) to bottom
    # right graph node (n,m)
    print(transition_mat)

    



if __name__=="__main__":
    print("testing")

    transmat = np.array([
        [np.inf,2,7,np.inf,np.inf],
        [np.inf,np.inf,3,8,5],
        [np.inf,2,np.inf,1,np.inf,],
        [np.inf,np.inf,np.inf,np.inf,4],
        [np.inf,np.inf,np.inf,5,np.inf,],
    ])
    start_node = 0
    end_node   = 24
    path_data = get_shortest_path(start_node,end_node,transmat)
    print("path_data:\n{}".format(path_data))