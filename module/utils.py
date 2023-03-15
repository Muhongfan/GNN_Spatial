import matplotlib.pyplot as plt
import pandas as pd
from processing import LoadData
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import h5py
import os
import random
import torch

def load_graphdata_channel(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, batch_size, shuffle=False):
    '''
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
    这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) +'_astcgn'

    print('load file:', filename)

    file_data = np.load(filename + '.npz')
    train_x = file_data['train_x']  # (10181, 307, 12, 9)
    train_x = train_x[:, :, 0:1,:]
    train_target = file_data['train_target']  # (10181, 307, 12)

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1,:]
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1,:]
    test_target = file_data['test_target']

    mean = file_data['mean'][:, :, 0:1,:]  # (1, 1, 1, 3)
    std = file_data['std'][:, :, 0:1,:]   # (1, 1, 1, 3)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers = 8
)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor)  #(B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 8
)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor) # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 8
)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std





class Evaluation(object):
    """
    Evaluation metrics
    """
    def __init__(self):
        pass
    @staticmethod
    def mae_(target, output):
        return mean_absolute_error(target, output)

    @staticmethod
    def mape_(target, output):
        return np.mean(np.abs(target-output)/ (target+5))

    @staticmethod
    def rmse_(target, output):
        return np.sqrt(mean_squared_error(target, output))

    @staticmethod
    def accuracy(target, output):
        """
        :param pred: predictions
        :param y: ground truth
        :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
        """
        return 1 - np.linalg.norm(target - output) / np.linalg.norm(target)

    @staticmethod
    def r2(target, output):
        """
        :param y: ground truth
        :param pred: predictions
        :return: R square (coefficient of determination)
        """
        return 1 - np.sum((target - output) ** 2) / np.sum((target - np.mean(output)) ** 2)

    @staticmethod
    def explained_variance(target, output):
        return 1 - np.var(target - output) / np.var(target)

    @staticmethod
    def total(target, output):
        mae = Evaluation.mae_(target,output)
        mape = Evaluation.mape_(target,output)
        rmse = Evaluation.rmse_(target,output)
        acc = Evaluation.accuracy(target, output)
        r2 = Evaluation.r2(target, output)
        explain_Var = Evaluation.explained_variance(target, output)
        return mae, mape, rmse, acc, r2, explain_Var

def compute_performance(prediction, target, data):
    """
    :param prediction: np.array, the predicted results
    :param target: np.array, the ground truth
    :param data: the test dataset
    :return: 
        performance: np.array, Evaluation metrics(MAE, MAPE, RMSE), 
        recovered_data: np.array, Recovered results
    """
    try:
        dataset = data.dataset
    except:
        dataset = data
    # print(len(dataset.flow_norm[0])) #207
    # print(len(prediction)) #512
    prediction = LoadData.recover_Data(dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy())
    target = LoadData.recover_Data(dataset.flow_norm[0], dataset.flow_norm[1], target.numpy())

    mae, mape, rmse, acc, r2, var = Evaluation.total(target.reshape(-1), prediction.reshape(-1))


    performance = [mae, mape, rmse, acc, r2, var]
    recovered_data = [prediction, target]

    return performance, recovered_data


def recover_data(prediction, target, feats, data):
    """
    :param prediction: np.array, the predicted results
    :param target: np.array, the ground truth
    :param data: the test dataset
    :return:
        performance: np.array, Evaluation metrics(MAE, MAPE, RMSE),
        recovered_data: np.array, Recovered results
    """
    try:
        dataset = data.dataset
    except:
        dataset = data
    # print(len(dataset.flow_norm[0])) #207
    # print(len(prediction)) #512
    prediction = LoadData.recover_Data(dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy())
    target = LoadData.recover_Data(dataset.flow_norm[0], dataset.flow_norm[1], target.numpy())

    mae, mape, rmse, acc, r2, var = Evaluation.total(target.reshape(-1), prediction.reshape(-1))


    performance = [mae, mape, rmse, acc, r2, var]
    recovered_data = [prediction, target]

    return performance, recovered_data


def save_feats(feat, path):
    # feats = []
    feats = feat.numpy()
    # feats = feats.tolist()


    # save_path = path +"feats.csv"
    np.savetxt('feats.csv', feats ,  delimiter=',')
    # feats_vec.to_csv(save_path, mode = 'a+', index = None, header = None)

def visualize_Result(path, h5_file, nodes_id, time_se, visualize_file):
    file = path + h5_file
    file_obj = h5py.File(file, "r")
    prediction = file_obj["predict"][:][:, :, 0] #[N, T]
    target = file_obj["target"][:][:, :, 0]
    file_obj.close()
    # save_path = path + '/Figure/'
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)
    
    plot_prediction = prediction[nodes_id][time_se[0]:time_se[1]]
    size_p = len(plot_prediction)   
    plot_target = target[nodes_id][time_se[0]:time_se[1]]
    size_t = len(plot_target)
    # visilization for a day
    plt.figure()
    plt.grid(True, linestyle="-.", linewidth = 0.5)
    plt.plot(np.array([t for t in range(time_se[1]-time_se[0])]), plot_prediction, ls="-", marker = " ", color = "r")
    plt.plot(np.array([t for t in range(time_se[1]-time_se[0])]), plot_target, ls="-", marker = " ", color = "b")
    plt.legend(["prediction", "target"], loc = "upper right")
    plt.axis([0, time_se[1]- time_se[0],
              np.min(np.array([np.min(plot_prediction), np.min(plot_target)])),
              np.max(np.array([np.max(plot_prediction), np.max(plot_target)]))])
    
    plt.savefig(path + visualize_file + ".png")
    plt.show()

    # plt.title("Training Loss")
    # plt.xlabel("time/5mins")
    # plt.ylabel("Traffic flow")
    # plt.plot(loss, label = 'Training_loss')
    # plt.legend()
    # plt.savefig(save_path + visualize_file + "_training loss.png" )
    # plt.show()
def plot_loss(path, visualize_file, train_loss, val_loss):

    #save_path = path + '/Figure/'
    plt.title("Training Loss")
    plt.xlabel("Time slot")
    plt.ylabel("Traffic flow")
    plt.plot(train_loss, label='Train_loss')
    plt.plot(val_loss, label='Test_loss')
    plt.legend()
    plt.savefig(path + visualize_file + "_ loss.png")
    plt.show()
    print("done")

def set_seed(seed = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


    
    