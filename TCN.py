import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import sys
sys.path.append("../../")



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """


        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """

        return self.network(x)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels):
        super(TCN, self).__init__()
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.tcn = TemporalConvNet(self.num_inputs, self.num_channels, kernel_size=3, dropout=0.2)
    def forward(self, data, device):
        flow_x = data["flow_x"].to(device)

        batch_size, num_nodes, seq_len, F = flow_x.size(0), flow_x.size(1), flow_x.size(2), flow_x.size(3)
        data = flow_x.reshape(batch_size, num_nodes * F, seq_len)
        output = self.tcn(data)

        # last time step
        output = output.reshape(batch_size, num_nodes, seq_len, F)[:, :, -1, :]
        return output.unsqueeze(2)

#
# class TCN(nn.Module):
#     def __init__(self, num_channels, num_feature, kernel_size=6, dropout=0.2):
#         super(TCN, self).__init__()
#         self._input_dim = num_feature  # 特征数
#         #self._hidden_dim = 1
#         layers = []
#         for i in range(len(num_channels)):
#             dilation_size = 2 ** i
#             in_channels = self._input_dim if i == 0 else num_channels[i - 1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
#
#         self.network = nn.Sequential(*layers)
#
#     def forward(self, data, device):
#         flow_x = data["flow_x"].to(device)
#         #graph = data["graph"].to(device)[0]  # [B, N, N]
#         # adj = GCN.get_adj(graph)
#
#         batch_size, num_nodes, seq_len, F = flow_x.size(0), flow_x.size(1), flow_x.size(2), flow_x.size(3)
#
#         #batch_size, seq_len, num_nodes = inputs.shape
#         #flow_x = flow_x.transpose(1, 2)
#         #flow_x = flow_x.view(batch_size, num_nodes, -1)  # [B, N, T * C]
#
#         data = flow_x.reshape(batch_size, num_nodes*F, seq_len)
#         output = self.network(data)
#         #output = output.unsqueeze(2)
#
#         # last time step
#         output = output.reshape(batch_size, num_nodes, seq_len, F)[:,:,-1,:]
#
#         return output.unsqueeze(2)


import torch.optim as optim
from torch.utils.data import DataLoader
from processing import LoadData

import time
if __name__ == '__main__':  # 测试模型是否合适
    # x = torch.randn(2, 6, 4, 1)  # [B, N, T, C]
    # graph = torch.randn(2, 6, 6)  # [N, N]
    # end_index = 3
    # data_y = x[:,:, end_index].unsqueeze(2)
    # data = {"flow_x": x, "graph": graph, "flow_y":data_y}
    # print(data_y.size())
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    #     print("running on the GPU")
    # else:
    #     device = torch.device("cpu")
    #     print("running on the CPU")
    # net = TCN(num_channels=[24,24], num_inputs= 6*1)
    # #net = GATNet2(input_dim=4, hidden_dim=2, output_dim=4, n_heads = 2,T = 4)
    # print(net)
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
    #     print(predict.size())
    #     print(data["flow_y"].size())
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

    net = TCN(num_channels=[307,307], num_inputs= 307)
    net = net.to(device)
    print(net)

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
            #print(predict.size())
            data_y = data["flow_y"]
            loss = criterion(predict, data_y)

            epoch_loss = loss.item()

            loss.backward()

            optimizer.step()  # 更新参数
        end_time = time.time()
        loss_train_plt.append(10 * epoch_loss / len(train_data) / 64)

        print("Epoch: {:04d}, Loss: {:02.4f}, TIme: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
                                                                          (end_time - start_time) / 60))

