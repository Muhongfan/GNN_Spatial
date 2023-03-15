import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from utils import save_feats
# import argparse
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import torch.optim as optim
# import time
# from processing import LoadData
# from torch.utils.data import DataLoader

# from utils.graph_conv import calculate_laplacian_with_self_loop


class GCN_model(nn.Module):
    def __init__(self, input_dim: int,  output_dim, **kwargs):
        super(GCN_model, self).__init__()
        # self.register_buffer(
        #     "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        # )
        # input_dim = T * C;
        # output_dim = T_p * C
        self._input_dim = input_dim  # seq_len for prediction
        #self._hidden_dim = hidden_dim # hidden_dim for prediction
        self._output_dim = output_dim
        self.weights = nn.Parameter(
            torch.FloatTensor(self._input_dim, self._output_dim)
        ) # T*C
        self.rule = torch.nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, data, device):
        # (batch_size, seq_len, num_nodes)
        # [B, N, T, C]
        flow_x = data["flow_x"].to(device)

        batch_size, num_nodes, seq_len, F = flow_x.size(0), flow_x.size(1), flow_x.size(2), flow_x.size(3)
        flow = flow_x.view(batch_size, num_nodes, -1)  # [B, N, T * C]
        graph = data["graph"].to(device)[0]
        laplacian = GCN_model.get_adj(graph, device)

        # [B, N, N] ,[B, N, T*C]-> AX [B, N, T*C]
        output_1 = torch.matmul(laplacian, flow)
        # AX [B, N, hid_c] -> AXW [B, N, out_dim] hid_c = T*C; out_dim = T_p * C
        outputs = torch.tanh(output_1 @ self.weights)
        # [B, N, 1, Out_C] , 就是 \hat AWX
        outputs = outputs.unsqueeze(2) #128, 207, 1, 1 B, N, T, C


        return outputs


    @staticmethod
    def get_adj(matrix, device):
        matrix = matrix + torch.eye(matrix.size(0)).to(device)
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
        return normalized_laplacian

class GCN_2model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim ): #kdropout, num_layers=2, return_embeds=False):
        super(GCN_2model, self).__init__()
        self.linear_1 = torch.nn.Linear(input_dim, hidden_dim)  # 定义一个线性层
        self.linear_2 = torch.nn.Linear(hidden_dim, output_dim)  # 定义一个线性层

        self.rule = torch.nn.ReLU()

    def forward(self, data, device):
        graph = data["graph"].to(device)[0]  # [B, N, N]
        adj = GCN_2model.get_adj(graph)

        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)

        # 第一个图卷积层
        output_1 = self.linear_1(flow_x)  # [B, N, hid_C],这个就是 WX，其中W是可学习的参数，X是输入的流量数据（就是flow_x）
        output_1 = self.rule(torch.matmul(adj, output_1))  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX

        output_2 = self.linear_2(output_1)

        output_2 = self.rule(torch.matmul(adj, output_2))

        return output_2

    @staticmethod
    def get_adj(graph):
        N = graph.size(0)
        matrix = torch.eye(N, dtype=torch.float, device=graph.device)
        graph += matrix  # A+I

        degree = torch.sum(graph, dim=1, keepdim=False)  # N
        degree = degree.pow(-1)  # -1 makes possibility for infinite
        degree[degree == float("inf")] = 0

        degree = torch.diag(degree)

        return torch.mm(degree, graph)  # AWX

class GCN(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.hidden_dim = hidden_dim

        self.gcn = GCN_model(self.input_dim, self.output_dim)
        # self.linear = nn.Linear(self.input_dim, 1)

    def forward(self, data, device):
        output = self.gcn(data, device)
        # output = self.linear(output)

        outputs = output
        return outputs
class Spa_feat(nn.Module):

    # input_dim = output_dim
    def __init__(self, input_dim, output_dim):
        super(Spa_feat, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gcn = GCN_model(self.input_dim, self.output_dim)

    def forward(self, data, device):
        spa_feat = self.gcn(data, device)

        return spa_feat



class GCN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        #self.gcn = GCN_model(self.input_dim, self.output_dim)
        self.gcn = GCN_2model(self.input_dim, self.hidden_dim, self.output_dim)

    def forward(self, data, device):
        output = self.gcn(data, device)

        outputs = output.unsqueeze(3)
        return outputs

class Spa2_feat(nn.Module):

    # input_dim = output_dim
    def __init__(self,  input_dim, hidden_dim, output_dim):
        super(Spa2_feat, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # self.gcn = GCN_model(self.input_dim, self.output_dim)
        self.gcn = GCN_2model(self.input_dim, self.hidden_dim, self.output_dim)

    def forward(self, data, device):
        spa_feats = self.gcn(data, device)
        spa_feat = spa_feats.unsqueeze(3)

        return spa_feat
