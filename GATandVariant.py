import torch
import torch.nn as nn
import torch.nn.functional as F
# from processing import LoadData
# from torch.utils.data import DataLoader


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = input_dim
        self.out_c = output_dim

        self.F = F.softmax

        self.W = nn.Linear(input_dim, output_dim, bias=False)  # y = W * x
        self.b = nn.Parameter(torch.Tensor(output_dim))

        nn.init.normal_(self.W.weight)
        #print(self.W.weight.size())
        nn.init.normal_(self.b)


    def forward(self, inputs, graph):
        """
        :param inputs: input features, [B, N, D].
        :param graph: graph structure, [N, N].
        :return:
            output features, [B, N, D].
        """

        h = self.W(inputs)  # [B, N, D]，一个线性层，就是第一步中公式的 W*h

        # 下面这个就是，第i个节点和第j个节点之间的特征做了一个内积，表示它们特征之间的关联强度
        # 再用graph也就是邻接矩阵相乘，因为邻接矩阵用0-1表示，0就表示两个节点之间没有边相连
        # 那么最终结果中的0就表示节点之间没有边相连

        outputs = torch.bmm(h, h.transpose(1, 2)) * graph.unsqueeze(0)  # [B, N, D]*[B, D, N]->[B, N, N],         x(i)^T * x(j)

        # 由于上面计算的结果中0表示节点之间没关系，所以将这些0换成负无穷大，因为softmax的负无穷大=0
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))

        attention = self.F(outputs, dim=2)  # [B, N, N]，在第２维做归一化，就是说所有有边相连的节点做一个归一化，得到了注意力系数
        result = torch.bmm(attention, h) + self.b
        return result  # [B, N, N] * [B, N, D]，，这个是第三步的，利用注意力系数对邻域节点进行有区别的信息聚合

class GAT_sin(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT_sin, self).__init__()
        self.attention_module = GraphAttentionLayer(input_dim, hidden_dim)# in_c为输入特征维度，hid_c为隐藏层特征维度
        self.out_att = GraphAttentionLayer(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, data, device):
        """
        :param inputs: [B, N, C]
        :param graph: [N, N]
        :return:
        """
        graph = data["graph"][0].to(device)  # [N, N]
        flow = data["flow_x"]  # [B, N, T, C]

        # flow = flow.transpose(0,1,3,2) #[ B, N, C, T]
        flow = flow.to(device)  # 将流量数据送入设备
        B, N = flow.size(0), flow.size(1)
        flow = flow.view(B, N, -1)  # [B, N, T * C]

        outputs = self.attention_module(flow, graph)  # [B, N, hid_c * h_head]
        outputs = self.act(outputs)
        outputs = self.out_att(outputs, graph)

        return outputs.unsqueeze(2)



class GAT_mul(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads):
        super(GAT_mul, self).__init__()

        self.attention_module = nn.ModuleList([GraphAttentionLayer(input_dim, hidden_dim) for _ in range(n_heads)])  # in_c为输入特征维度，hid_c为隐藏层特征维度

        # 上面的多头注意力都得到了不一样的结果，使用注意力层给聚合起来
        self.out_att = GraphAttentionLayer(hidden_dim * n_heads, output_dim)

        self.act = nn.ReLU()


    def forward(self, inputs, graph):
        """
        :param inputs: [B, N, C]
        :param graph: [N, N]
        :return:
        """
        # 每一个注意力头用循环取出来，放入list里，然后在最后一维串联起来
        outputs = torch.cat([attn(inputs, graph) for attn in self.attention_module], dim=-1)  # [B, N, hid_c * h_head]
        outputs = self.act(outputs)
        outputs = self.out_att(outputs, graph)

        return self.act(outputs)


class GATNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads):
        super(GATNet, self).__init__()
        self.subnet = GAT_mul(input_dim, hidden_dim, output_dim, n_heads)

    def forward(self, data, device):
        graph = data["graph"][0].to(device)  # [N, N]
        flow = data["flow_x"]  # [B, N, T, C]


        #flow = flow.transpose(0,1,3,2) #[ B, N, C, T]
        flow = flow.to(device)  # 将流量数据送入设备
        B, N = flow.size(0), flow.size(1)
        flow = flow.view(B, N, -1)  # [B, N, T * C]

        """
       上面是将这一段的时间的特征数据摊平做为特征，这种做法实际上忽略了时序上的连续性
       这种做法可行，但是比较粗糙，当然也可以这么做：
       flow[:, :, 0] ... flow[:, :, T-1]   则就有T个[B, N, C]这样的张量，也就是 [B, N, C]*T
       每一个张量都用一个SubNet来表示，则一共有T个SubNet，初始化定义　self.subnet = [GATSubNet(...) for _ in range(T)]
       然后用nn.ModuleList将SubNet分别拎出来处理，参考多头注意力的处理，同理
       """

        prediction = self.subnet(flow, graph).unsqueeze(2)  # [B, N, 1, C]，这个１加上就表示预测的是未来一个时刻
        return prediction

class GATNet2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, T):
        super(GATNet2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.T = T 

        self.gatnet = nn.ModuleList([GAT_mul(self.input_dim, self.hidden_dim, self.output_dim, self.n_heads) for _ in range(self.T)])
        self.out = GraphAttentionLayer(self.output_dim*self.T, self.output_dim)


    def forward(self, data, device):
        flow = data["flow_x"].to(device)
        graph = data["graph"][0].to(device)
        T = flow.size(2)

        output = []
        
        for i in range(T):
            f = flow[:,:,i,:]
            output.append(self.gatnet[i](f, graph))

            #hidden_put = torch.cat((hidden_put,self.gatnet[i](f, graph)), dim=2)
            #output.append(gat(f, graph) for gat in self.gat) torch.cat([gat(f, graph) for gat in self.gat], dim =2)
        #print(output)
        
        hidden_put = torch.cat([i for i in output], dim=2)
        #print(hidden_put)
        #print(hidden_put.size())
        out = self.out(hidden_put, graph)
        out = out.unsqueeze(2)
        #print(out.size())

        return out

import torch.optim as optim
import time
if __name__ == '__main__':  # 测试模型是否合适
    x = torch.randn(2, 6, 4, 4)  # [B, N, T, C]
    graph = torch.randn(2, 6, 6)  # [N, N]
    end_index = 3
    data_y = x[:,:, end_index].unsqueeze(2)
    data = {"flow_x": x, "graph": graph, "flow_y":data_y}
    print(data_y.size())
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("running on the GPU")
    else:
        device = torch.device("cpu")
        print("running on the CPU")
    net = GATNet(input_dim=4 * 4, hidden_dim=2, output_dim=4, n_heads = 2)
    #net = GATNet2(input_dim=4, hidden_dim=2, output_dim=4, n_heads = 2,T = 4)
    print(net)
    net = net.to(device)

    #y = net(dataset, device)
    #print(y.size())

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

        net.zero_grad()
        predict = net(data, device).to(torch.device("cpu"))
        print(predict.size())
        loss = criterion(predict, data["flow_y"])

        epoch_loss = loss.item()

        loss.backward()
        optimizer.step()  # 更新参数
        end_time = time.time()
        loss_train_plt.append(10 * epoch_loss / len(data) / 64)

        print("Epoch: {:04d}, Loss: {:02.4f}, TIme: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(data),
                                                                          (end_time - start_time) / 60))

    # train_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
    #                       time_interval=5, history_length=6,
    #                       train_mode="train")
    # 
    # train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=8)
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    #     print("running on the GPU")
    # else:
    #     device = torch.device("cpu")
    #     print("running on the CPU")
    # 
    # net = GATNet(input_dim=6 * 1, hidden_dim=3, output_dim=2, n_heads = 2)
    # net = net.to(device)
    # 
    # for data in train_loader:
    #     print(data.keys())
    #     print(data['graph'].size())
    #     print(data['flow_x'].size())
    #     print(data['flow_y'].size())
    #     y = net(data, device)
    # print(y.size())


# #%%
# import numpy as np
# file = 'PeMS_04/PeMS04.npz'
# p = np.load(file, allow_pickle = True)
# print(p.files)
# print(len(p['data'][0]))



