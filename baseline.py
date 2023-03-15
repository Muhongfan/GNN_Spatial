import torch
import torch.nn as nn

class GCN3(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim ): #kdropout, num_layers=2, return_embeds=False):
        super(GCN3, self).__init__()
        self.linear_1 = torch.nn.Linear(input_dim, hidden_dim)  # 定义一个线性层
        self.linear_2 = torch.nn.Linear(hidden_dim, hidden_dim)  # 定义一个线性层
        self.linear_3 = torch.nn.Linear(hidden_dim, output_dim)  # 定义一个线性层
        self.rule = torch.nn.ReLU()

    def forward(self, data, device):
        graph = data["graph"].to(device)[0]  # [B, N, N]
        adj = GCN3.get_adj(graph)

        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)

        # 第一个图卷积层
        output_1 = self.linear_1(flow_x)  # [B, N, hid_C],这个就是 WX，其中W是可学习的参数，X是输入的流量数据（就是flow_x）
        output_1 = self.rule(torch.matmul(adj, output_1))  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX

        output_2 = self.linear_3(output_1)
        output_2 = self.rule(torch.matmul(adj, output_2))

        output_3 = self.linear_3(output_2)
        output_3 = self.rule(torch.matmul(adj, output_3))


        return output_3.unsqueeze(2)  # [B, N, 1, Out_C] , 就是 \hat AWX

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



class GCN2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim ): #kdropout, num_layers=2, return_embeds=False):
        super(GCN2, self).__init__()
        self.linear_1 = torch.nn.Linear(input_dim, hidden_dim)  # 定义一个线性层
        self.linear_2 = torch.nn.Linear(hidden_dim, output_dim)  # 定义一个线性层

        self.rule = torch.nn.ReLU()

    def forward(self, data, device):
        graph = data["graph"].to(device)[0]  # [B, N, N]
        adj = GCN2.get_adj(graph)

        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)

        # 第一个图卷积层
        output_1 = self.linear_1(flow_x)  # [B, N, hid_C],这个就是 WX，其中W是可学习的参数，X是输入的流量数据（就是flow_x）
        output_1 = self.rule(torch.matmul(adj, output_1))  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX

        output_2 = self.linear_2(output_1)
        output_2 = self.rule(torch.matmul(adj, output_2))  

        return output_2.unsqueeze(2)  # [B, N, 1, Out_C] , 就是 \hat AWX

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

class bGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim ): #kdropout, num_layers=2, return_embeds=False):
        super(bGCN, self).__init__()
        self.linear_1 = torch.nn.Linear(input_dim, hidden_dim)  # 定义一个线性层
        self.linear_2 = torch.nn.Linear(hidden_dim, output_dim)  # 定义一个线性层

        self.rule = torch.nn.ReLU()

    def forward(self, data, device):
        graph = data["graph"].to(device)[0]  # [B, N, N]
        adj = bGCN.get_adj(graph)

        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)

        # 第一个图卷积层
        output_1 = self.linear_1(flow_x)  # [B, N, hid_C],这个就是 WX，其中W是可学习的参数，X是输入的流量数据（就是flow_x）
        output_1 = self.rule(torch.matmul(adj, output_1))  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX

        # output_2 = self.linear_2(output_1)
        # output_2 = self.rule(torch.matmul(adj, output_2))

        return output_1.unsqueeze(2)  # [B, N, 1, Out_C] , 就是 \hat AWX

    @staticmethod
    def get_adj(graph):
        N = graph.size(0)
        matrix = torch.eye(N, dtype=torch.float, device=graph.device)
        graph += matrix  # A+I

        degree = torch.sum(graph, dim=1, keepdim=False)  # N
        degree = degree.pow(-0.5)  # -1 makes possibility for infinite
        #degree[degree == float("inf")] = 0
        degree[torch.isinf(degree)] = 0.0
        degree = torch.diag(degree)
        normalized_laplacian = (
            matrix.matmul(degree).transpose(0, 1).matmul(degree)
        )

        return normalized_laplacian  # AWX


# x * g_/theta = U_g_/theta U^T x = /sum{k=0, K} /alpha_k T_k (L) x
class ChebConv(torch.nn.Module):
    """
    THE ChebNet convolution operation

    input_dim: int

    output_dim: int
    """

    def __init__(self, input_dim, output_dim, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize
        self.weight = torch.nn.Parameter(torch.Tensor(K + 1, 1, input_dim, output_dim))  # /alpha_k [K+1, 1, in, out]
        torch.nn.init.xavier_normal_(self.weight)  # /alpha_k

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1, 1, output_dim))
            torch.nn.init.zeros_(self.bias)  # filling with 0
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, A):
        """
        :param inputs: [B, N, C]
        :param graph: adj matrix [N, N]
        :return: [B, N, D]
        """

        L = ChebConv.get_laplacian(A, self.normalize)
        mul_L = self.cheb_polynomial(L).unsqueeze(1)  # [K, 1, N, N]，

        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]
        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]
        return result

    def cheb_polynomial(self, laplacian):
        """
        :param laplacian: [N, N]
        :return: [K, N, N]
        """
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - \
                                               multi_order_laplacian[k - 2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(A, normalize):  # 计算拉普拉斯矩阵
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(A, dim=-1) ** (-1 / 2))  # 这里的graph就是邻接矩阵,这个D
            L = torch.eye(A.size(0), device=A.device, dtype=A.dtype) - torch.mm(torch.mm(D, A),
                                                                                D)  # L = I - D * A * D,这个也就是正则化
        else:
            D = torch.diag(torch.sum(A, dim=-1))
            L = D - A
        return L


class ChebNet(torch.nn.Module):  # 定义图网络的类
    def __init__(self, input_dim, hidden_dim, output_dim, K):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.class
        :param out_c: int, number of output channels.
        :param K:
        """
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(input_dim=input_dim, output_dim=hidden_dim, K=K)  # 第一个图卷积层
        self.conv2 = ChebConv(input_dim=hidden_dim, output_dim=output_dim, K=K)  # 第二个图卷积层
        self.rule = torch.nn.ReLU()  # 激活函数

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N]
        flow_x = data["flow_x"].to(device)  # [B, N, H, D]  # B是batch size，N是节点数，H是历史数据长度，D是特征维度

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D] H = 6, D = 1把最后两维缩减到一起了，这个就是把历史时间的特征放一起

        output_1 = self.rule(self.conv1(flow_x, graph_data))
        output_2 = self.rule(self.conv2(output_1, graph_data))

        return output_2.unsqueeze(3)


import torch.nn.functional as F


class GraphAttentionLayer(torch.nn.Module):
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


class GATSubNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads):
        super(GATSubNet, self).__init__()

        self.attention_module = nn.ModuleList([GraphAttentionLayer(input_dim, hidden_dim,) for _ in range(n_heads)])  # in_c为输入特征维度，hid_c为隐藏层特征维度

        # 上面的多头注意力都得到了不一样的结果，使用注意力层给聚合起来
        self.out_att = GraphAttentionLayer(hidden_dim * n_heads, output_dim)

        self.act = nn.LeakyReLU()


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


class GATNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads):
        super(GATNet, self).__init__()
        self.subnet = GATSubNet(input_dim, hidden_dim,  output_dim, n_heads)

    def forward(self, data, device):
        graph = data["graph"][0].to(device)  # [N, N]
        flow = data["flow_x"]  # [B, N, T, C]
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

        self.gatnet = nn.ModuleList([GATSubNet(input_dim, hidden_dim, output_dim, n_heads) for _ in range(T)])
        self.out = GraphAttentionLayer(output_dim * T, output_dim)

    def forward(self, data, device):
        flow = data["flow_x"].to(device)
        graph = data["graph"][0].to(device)
        T = flow.size(2)

        output = []

        for i in range(T):
            f = flow[:, :, i, :]
            output.append(self.gatnet[i](f, graph))

            # hidden_put = torch.cat((hidden_put,self.gatnet[i](f, graph)), dim=2)
            # output.append(gat(f, graph) for gat in self.gat) torch.cat([gat(f, graph) for gat in self.gat], dim =2)
        # print(output)

        hidden_put = torch.cat([i for i in output], dim=2)
        # print(hidden_put)
        # print(hidden_put.size())
        out = self.out(hidden_put, graph)
        out = out.unsqueeze(2)
        # print(out.size())

        return out


import torch.optim as optim
import time

if __name__ == '__main__':  # 测试模型是否合适
    x = torch.randn(2, 6, 4, 4)  # [B, N, T, C]
    graph = torch.randn(2, 6, 6)  # [N, N]
    end_index = 3
    data_y = x[:,:, end_index].unsqueeze(2)
    print(data_y[0])
    data = {"flow_x": x, "graph": graph, "flow_y":data_y}
    print(data_y.size())
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("running on the GPU")
    else:
        device = torch.device("cpu")
        print("running on the CPU")
    net = GCN(input_dim=4 * 1, hidden_dim=2, output_dim=1)
    # net = GATNet2(input_dim=4, hidden_dim=2, output_dim=4, n_heads = 2,T = 4)
    print(net)
    net = net.to(device)

    # y = net(dataset, device)
    # print(y.size())

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
    # net = GATNet(input_dim=6 * 2, hidden_dim=6, output_dim=2, n_heads=2)
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


# if __name__ == '__main__':  # 测试模型是否合适
#     x = torch.randn(32, 278, 6, 2)  # [B, N, T, C]
#     graph = torch.randn(32, 278, 278)  # [N, N]
#     data = {"flow_x": x, "graph": graph}
# 
#     device = torch.device("cpu")
#     # tensor = torch.to(device)
# 
#     net = GATNet(input_dim=6 * 2, hidden_dim=6, output_dim=2, n_heads=2)
# 
#     y = net(data, device)
#     print(y.size())
