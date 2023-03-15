import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from TCN import TemporalConvNet

import sys

sys.path.append("../../")

class GCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super(GCN, self).__init__()
        # self.register_buffer(
        #     "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        # )

        self._input_dim = input_dim  # seq_len for prediction
        self._output_dim = output_dim  # hidden_dim for prediction
        self.weights = nn.Parameter(
            torch.FloatTensor(self._input_dim, self._output_dim)
        )
        self.rule = torch.nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))


    # tanhdef forward(self, data, device):
    #     # (batch_size, seq_len, num_nodes)
    #     # [B, N, T, C]
    #     flow_x = data["flow_x"].to(device)
    #
    #     batch_size, num_nodes, seq_len, F = flow_x.size(0), flow_x.size(1), flow_x.size(2), flow_x.size(3)
    #     flow = flow_x.view(batch_size, num_nodes, -1)  # [B, N, T * C]
    #     graph = data["graph"].to(device)[0]

    def forward(self, flow_x, graph, device):
        # (batch_size, seq_len, num_nodes)
        # [B, N, T, C]

        batch_size, num_nodes, seq_len, F = flow_x.size(0), flow_x.size(1), flow_x.size(2), flow_x.size(3)
        flow = flow_x.view(batch_size, num_nodes, -1)  # [B, N, T * C]


        laplacian = GCN.get_adj(graph, device)

        # [B, N, N] ,[B, N, hid_c]-> AX [B, N, hid_c]
        output_1 = torch.matmul(laplacian, flow)
        # AX [B, N, hid_c] -> AXW [B, N, out_dim]
        outputs = torch.tanh( torch.matmul(output_1,  self.weights) )
        # [B, N, , Out_C] , 就是 \hat AWX

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

class GCTCN(nn.Module):
    def __init__(self, num_inputs, num_channels, input_dim, output_dim):
        super(GCTCN, self).__init__()

        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tcn = TemporalConvNet(self.num_inputs, self.num_channels, kernel_size=2, dropout=0.2)
        self.gcn = GCN(self.input_dim, self.output_dim)

    def forward(self, data, device):
        flow_x = data["flow_x"].to(device)

        # graph = data["graph"].to(device)[0]
        # output_1 = self.gcn(flow_x, graph, device)
        # output_1 = output_1.unsqueeze(3)
        #
        # batch_size, num_nodes, seq_len, F = output_1.size(0), output_1.size(1), output_1.size(2), output_1.size(3)
        # #output_1 = output_1.transpose(2, 1)
        # data = output_1.reshape(batch_size , num_nodes * F, seq_len)
        # output_2 = self.tcn(data)
        # # last time step
        # output_2 = output_2.reshape(batch_size, num_nodes, F, seq_len)
        # output_2 = output_2.transpose(2, 3)

        graph = data["graph"].to(device)[0]
        output_1 = self.gcn(flow_x, graph, device)
        # output_1 = output_1.unsqueeze(3)
        # 
        batch_size, num_nodes, seq_len, F = flow_x.size(0), flow_x.size(1), flow_x.size(2), flow_x.size(3)
        # #output_1 = output_1.transpose(2, 1)
        # data = output_1.reshape(batch_size , num_nodes * F, seq_len)
        output_2 = self.tcn(output_1)
        # last time step

        output_2 = output_2.reshape(batch_size, num_nodes, seq_len ,F)[:, :, -1, :]
        output_2 = output_2.unsqueeze(2)

        return output_2


import torch.optim as optim
import time
from torch.utils.data import DataLoader
from processing import LoadData

if __name__ == '__main__':  # 测试模型是否合适
    # x = torch.randn(2, 6, 4, 1)  # [B, N, T, C]
    # graph = torch.randn(2, 6, 6)  # [N, N]
    # end_index = 3
    # data_y = x[:,:, end_index].unsqueeze(2)
    # data = {"flow_x": x, "graph": graph, "flow_y":data_y}
    # #print(data_y.size())
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    #     print("running on the GPU")
    # else:
    #     device = torch.device("cpu")
    #     print("running on the CPU")
    # net = GCTCN(num_channels=[6, 6], num_inputs=6*1, input_dim=4*1 , output_dim=1)
    # #net = GATNet2(input_dim=4, hidden_dim=2, output_dim=4, n_heads = 2,T = 4)
    # #print(net)
    # net = net.to(device)
    #
    # #y = net(dataset, device)
    # #print(y.size())
    #
    # # loss and optimizer
    # criterion = torch.nn.MSELoss()
    # optimizer = optim.Adam(params=net.parameters())
    #
    # # train
    # Epoch = 10
    # loss_train_plt = []
    #
    # net.train()
    # for epoch in range(Epoch):
    #     epoch_loss = 0.0
    #     start_time = time.time()
    #
    #     net.zero_grad()
    #     predict = net(data, device).to(torch.device("cpu"))
    #     #print(data["flow_y"].size())
    #     loss = criterion(predict, data["flow_y"])
    #
    #     epoch_loss = loss.item()
    #
    #     loss.backward()
    #     optimizer.step()  # 更新参数
    #     end_time = time.time()
    #     loss_train_plt.append(10 * epoch_loss / len(data) / 64)
    #
    #     print("Epoch: {:04d}, Loss: {:02.4f}, TIme: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(data),
    #                                                                       (end_time - start_time) / 60))

    train_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                          time_interval=5, history_length=6,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=8)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("running on the GPU")
    else:
        device = torch.device("cpu")
        print("running on the CPU")


    net = GCTCN(num_channels=[307, 307], num_inputs=307 * 1, input_dim=6, output_dim=1)

    #net = GCTCN(num_channels=[307, 307], num_inputs=307 * 1, input_dim=6, output_dim=6)

    net = net.to(device)

    # for data in train_loader:
    #     # print(data.keys())
    #     # print(data['graph'].size())
    #     # print(data['flow_x'].size())
    #     # print(data['flow_y'].size())
    #     y = net(data, device)
    #     # print(y.size())
    # loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(params=net.parameters())

    # train
    Epoch = 10
    loss_train_plt = []

    net.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()
        for data in train_loader:
            net.zero_grad()
            predict = net(data, device).to(torch.device("cpu"))
            # print(predict.size())
            data_y = data["flow_y"]
            loss = criterion(predict, data_y)

            epoch_loss = loss.item()

            loss.backward()

            optimizer.step()  # 更新参数
        end_time = time.time()
        loss_train_plt.append(10 * epoch_loss / len(train_data) / 64)

        print("Epoch: {:04d}, Loss: {:02.4f}, TIme: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
                                                                          (end_time - start_time) / 60))

