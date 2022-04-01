# Created by Yinuo
# This file is currently used to split a molecular system box into two parts with some buffer space
# The idea is spliting from the middle of longest box edge, then add 5.2A buffer space for each sub_box or group
# For usage, please call function divide_box


import torch


def _split_on_longest_edge(edge_list, edge_length, cutoff):
    '''
    This function is used to split the box based on the longest edge
    edge_middle: find the edge's middle point coord
    Input:
          edge_list: 1d torch tensor for the longest edge dimension (x or y or z)
          edge_length: length of this edge
          cutoff: cutoff of aev
    Ouput:
          G1: atom index of Group1 (with cutoff buffer)
          G1_core: atom index of the core of Group1 (without cutoff buffer)
          G2: atom index of Group2 (with cutoff buffer)
          G2_core: atom index of the core of Group2 (without cutoff buffer)
          others are the size of G1_id, G1_core_id, G2_id, G2_core_id
    '''
    edge_middle = edge_length / 2
    G1 = (edge_list < edge_middle + cutoff).nonzero().squeeze()
    G2 = (edge_list > edge_middle - cutoff).nonzero().squeeze()
    G1_core = (edge_list < edge_middle).nonzero().squeeze()
    G2_core = (edge_list >= edge_middle).nonzero().squeeze()
    return G1, G1_core, G2, G2_core


def divide_box(species, coords, cell, cutoff):
    '''
    This funciton is used to divide the input atoms box into two groups with a buffer space of cutoff(5.2A)
    The usage buffer space is to ensure the correctness of AEV computation when splitting the box from middle
    Idea: compute length/width/height of the box, split the box from the middle of longest edge, generate two sub_boxes
          G1 = sub_box1 + 5.2A buffer space (G1 can be used to generate )
          G1_core = sub_box1
          G2 = sub_box2 + 5.2A buffer space
          G2_core = sub_box2
    Input:
          species: species torch tensor, dim = 2
          coords: coordinates torch tensor, dim = 3
          cutoff: aev cutoff, the default value is 5.2A
    Output:
          G1_species, G1_coords
          G1_core_species, G1_core_coords
          G2_species, G2_coords
          G2_core_species, G2_core_coords
    '''

    # find the max and min values for each dimension
    diagonal = torch.linalg.norm(cell, dim=0)
    longest_edge = torch.argmax(diagonal)
    G1, G1_core, G2, G2_core = _split_on_longest_edge(coords[0][:, longest_edge], diagonal[longest_edge], cutoff)

    G1_species = species[:, G1]
    G1_core_species = species[:, G1_core]
    G2_species = species[:, G2]
    G2_core_species = species[:, G2_core]

    G1_coords = coords[:, G1]
    G1_core_coords = coords[:, G1_core]
    G2_coords = coords[:, G2]
    G2_core_coords = coords[:, G2_core]
    return G1_species, G1_core_species, G2_species, G2_core_species, G1_coords, G1_core_coords, G2_coords, G2_core_coords
