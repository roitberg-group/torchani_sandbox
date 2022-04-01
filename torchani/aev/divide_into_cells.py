# Created by Yinuo
# This file is currently used to split a molecular system box into two parts with some buffer space
# The idea is spliting from the middle of longest box edge, then add 5.2A buffer space for each sub_box or group
# For usage, please call function divide_box


import torch


def split_on_longest_edge(edge_list, edge_min, edge_length, cutoff):
    '''
    This function is used to split the box based on the longest edge
    edge_middle: find the edge's middle point coord
    Input:
          edge_list: 1d torch tensor for the longest edge dimension (x or y or z)
          edge_min: smallest coord
          edge_length: length of this edge
          cutoff: cutoff of aev
    Ouput:
          G1_id: atom index of Group1 (with cutoff buffer)
          G1_core_id: atom index of the core of Group1 (without cutoff buffer)
          G2_id: atom index of Group2 (with cutoff buffer)
          G2_core_id: atom index of the core of Group2 (without cutoff buffer)
          others are the size of G1_id, G1_core_id, G2_id, G2_core_id
    '''
    edge_middle = edge_min + edge_length/2
    G1_id = []
    G1_core_id = []
    G2_id = []
    G2_core_id = []
    
    for i, edge_coord in enumerate(edge_list):
        
        if edge_coord < edge_middle + cutoff:
            G1_id.append(i)
            if edge_coord <= edge_middle:
                G1_core_id.append(i)
                
                
        if edge_coord > edge_middle - cutoff:
            G2_id.append(i)
            if edge_coord > edge_middle:
                G2_core_id.append(i)
                
    return G1_id, len(G1_id), G1_core_id, len(G1_core_id), G2_id, len(G2_id), G2_core_id, len(G2_core_id),



def divide_box(spc, coords, cutoff):
    '''
    This funciton is used to divide the input atoms box into two groups with a buffer space of cutoff(5.2A)
    The usage buffer space is to ensure the correctness of AEV computation when splitting the box from middle 
    Idea: compute length/width/height of the box, split the box from the middle of longest edge, generate two sub_boxes
          G1 = sub_box1 + 5.2A buffer space (G1 can be used to generate )
          G1_core = sub_box1
          G2 = sub_box2 + 5.2A buffer space
          G2_core = sub_box2
    Input:
          spc: species torch tensor, dim = 2
          coords: coordinates torch tensor, dim = 3
          cutoff: aev cutoff, the default value is 5.2A
    Output:
          G1_spc, G1_coords
          G1_core_spc, G1_core_coords
          G2_spc, G2_coords
          G2_core_spc, G2_core_coords
    '''
    
    # find the x, y, z coords in all atoms
    xlist = coords[0][:,0]
    ylist = coords[0][:,1]
    zlist = coords[0][:,2]
    
    # find the max and min values for each dimension
    x_max = torch.max(xlist)
    x_min = torch.min(xlist)
    x_len = x_max - x_min
    y_max = torch.max(ylist) 
    y_min = torch.min(ylist)
    y_len = y_max - y_min
    z_max = torch.max(zlist) 
    z_min = torch.min(zlist)
    z_len = z_max - z_min

    # find out the longest edge and split the box into two groups
    if x_len >= y_len and x_len >= z_len:
        # x is the widest dimension, split the box from x edge
        G1_id, G1_len, G1_core_id, G1_core_len, G2_id, G2_len, G2_core_id, G2_core_len = split_on_longest_edge(xlist, x_min, x_len, cutoff)
    elif y_len >= x_len and y_len >= z_len:
        # y is the widest dimension, split the box from y edge
        G1_id, G1_len, G1_core_id, G1_core_len, G2_id, G2_len, G2_core_id, G2_core_len = split_on_longest_edge(ylist, y_min, y_len, cutoff)
    else:
        # z is the widest dimension, split the box from z edge
        G1_id, G1_len, G1_core_id, G1_core_len, G2_id, G2_len, G2_core_id, G2_core_len = split_on_longest_edge(zlist, z_min, z_len, cutoff)

    # create empty tensors for two groups and their core groups
    G1_spc = torch.empty((1, G1_len), dtype=int)
    G1_coords = torch.empty((1,G1_len,3), dtype=torch.float32)
    G1_core_spc = torch.empty((1, G1_core_len), dtype=int)
    G1_core_coords = torch.empty((1,G1_core_len,3), dtype=torch.float32)
    G2_spc = torch.empty((1, G2_len), dtype=int)
    G2_coords = torch.empty((1,G2_len,3), dtype=torch.float32)
    G2_core_spc = torch.empty((1, G2_core_len), dtype=int)
    G2_core_coords = torch.empty((1,G2_core_len,3), dtype=torch.float32)
    
    # assign spc and coords to each group
    for i in range(G1_len):
        index = G1_id[i]
        G1_spc[0][i] = spc[0][index]
        G1_coords[0][i] = coords[0][index]

    for i in range(G1_core_len):
        index = G1_core_id[i]
        G1_core_spc[0][i] = spc[0][index]
        G1_core_coords[0][i] = coords[0][index]

    for i in range(G2_len):
        index = G2_id[i]
        G2_spc[0][i] = spc[0][index]
        G2_coords[0][i] = coords[0][index]
        
    for i in range(G2_core_len):
        index = G2_core_id[i]
        G2_core_spc[0][i] = spc[0][index]
        G2_core_coords[0][i] = coords[0][index]
    
    return G1_spc, G1_coords, G1_core_spc, G1_core_coords, G2_spc, G2_coords, G2_core_spc, G2_core_coords

