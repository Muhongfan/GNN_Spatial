import csv
import torch
import numpy as np
from torch.utils.data import Dataset

def get_adjacent_matrix(distance_file, num_nodes, id_file = None, graph_type = "connect"):
    A = np.zeros([int(num_nodes), int(num_nodes)])
    
    if id_file: 
        with open(id_file, 'r') as f_id:
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split("\n"))}
            
            with open(distance_file, "r") as f_d:
                f_d.readline()  # jump the header
                reader = csv.reader(f_d)
                for item in reader:
                    if len(item) !=3:
                        continue
                    i, j, distance = int(item[0]), int(item[1]), float(item[2])
                    if graph_type == "connect":
                        A[node_id_dict[i], node_id_dict[j]] = 1
                        A[node_id_dict[j], node_id_dict[i]] = 1
                    elif graph_type == "distance":
                        A[node_id_dict[i], node_id_dict[j]] = 1./distance
                        A[node_id_dict[j], node_id_dict[i]] = 1./diatance
                    else:
                        raise ValueError("Graph can not be constructed as the type not be conveyed")
        return A
    
    with open(distance_file, "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)
        for item in reader:
            if len(item) !=3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])
            if graph_type == "connect":
                A[i, j], A[j, i] =1., 1.

            elif graph_type == "distance":
                A[i, j] = 1. / distance
                A[j, i] = 1. / diatance
            else:
                raise ValueError("Graph can not be constructed as the type not be conveyed")
        return A
                        
    