
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, accuracy_score, mean_squared_error, \
    r2_score, explained_variance_score

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from baseline import ChebNet
from set_seed import set_seed
from train import train, evaluate
from Initialize_model import initialize_model
from test import test
from processing import LoadData
# fromutils import Evaluation
from utils import visualize_Result
from utils import compute_performance, set_seed
from GATandVariant import GAT_sin
from base_model import GCN, GCN2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from save_file import save_file
import argparse
import configparser

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='/home/amber/Documents/Feat_fusion/config/PEMS04.conf', type=str,
                    help="configuration file path")
# parser.add_argument("--config", default='/home/amber/Documents/Feat_fusion/config/Losloop.conf', type=str,
#                     help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config,  encoding="utf-8")

training_config = config['default']
dataset = training_config['dataset']
nums_of_nodes = int(training_config['nums_of_nodes'])
time_interval = int(training_config['time_interval'])
#history_len = int(training_config['len'])
history_len = config.get('default', 'len')
batch_size = int(training_config['batch_size'])
learning_rate = float(training_config['learning_rate'])
#model_name = training_config['model_name']
epochs = int(training_config['epochs'])
hidden_dim = int(training_config['hidden_dim'])
training_days = int(training_config['training_days'])
validation_days = int(training_config['validation_days'])
test_days = int(training_config['test_days'])
# task = config.get('default', 'task')

# slot = history_len * time_interval
devide_days = [training_days, validation_days, test_days]

training_data = config.get('default', 'training_data')
testing_data = config.get('default', 'testing_data')
data_path = [training_data, testing_data]
# slot=404

torch.cuda.empty_cache()
for his_len in str.split(history_len):
    his_len = int(his_len)
    slot = int(his_len) * time_interval

    # train_data = LoadData(data_path = ["../data/Los/los_adj.csv", "../data/Los/los_speed.csv"], num_nodes = nums_of_nodes, divide_days =devide_days,
    #                           time_interval = time_interval, history_length=his_len, train_mode = "train")
    # train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = False, num_workers = 8)
    # valid_data = LoadData(data_path = ["../data/Los/los_adj.csv", "../data/Los/los_speed.csv"], num_nodes = nums_of_nodes, divide_days =devide_days,
    #                           time_interval =time_interval, history_length=his_len, train_mode = "valid")
    # valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = False, num_workers = 8)
    # test_data = LoadData(data_path =  ["../data/Los/los_adj.csv", "../data/Los/los_speed.csv"], num_nodes = nums_of_nodes, divide_days =devide_days,
    #                           time_interval = time_interval, history_length=his_len, train_mode = "test")
    # test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, num_workers = 8)
    train_data = LoadData(data_path = data_path, num_nodes = nums_of_nodes, divide_days =devide_days,
                              time_interval = time_interval, history_length=his_len, train_mode = "train")
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = False, num_workers = 8)
    valid_data = LoadData(data_path = data_path, num_nodes = nums_of_nodes, divide_days =devide_days,
                              time_interval =time_interval, history_length=his_len, train_mode = "valid")
    valid_loader = DataLoader(train_data, batch_size = batch_size, shuffle = False, num_workers = 8)
    test_data = LoadData(data_path =  data_path, num_nodes = nums_of_nodes, divide_days =devide_days,
                              time_interval = time_interval, history_length=his_len, train_mode = "test")
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, num_workers = 8)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("running on the GPU")
    else:
        device = torch.device("cpu")
        print("running on the CPU")


    def training(model):
        torch.cuda.empty_cache()

        # For prediction
        set_seed(42)  # Set seed for reproducibility
        model, criterion, optimizer, scheduler = initialize_model(model, learning_rate, device)
        path = save_file(dataset, slot, model)
        print("finished initialization")
        loss_train_plt, loss_val_plt, best_loss= train(model, device, train_loader, valid_loader, criterion, optimizer, scheduler,
                                             epochs, nums_of_nodes, path, evaluation=True)
        # print(loss_train_plt)
        # print("finished trained")
        acc = test(model, device, criterion, test_loader, loss_train_plt, loss_val_plt, path, nums_of_nodes, slot)

        return acc


    print("GCN:")
    model = GCN(input_dim=his_len * 1, output_dim=1)

    # print("GCN2:")
    # model = GCN2(input_dim=his_len, hidden_dim=nums_of_nodes, output_dim=1)

    # print("GAT:")
    # model = GAT_sin(input_dim=his_len, hidden_dim = 3, output_dim= 1)

    # print("GRU:")
    # model = GRU(hidden_dim=hidden_dim * 1, output_dim= 1)

    # training(model)

    # print("GAT:")
    # train(GATNet(input_dim=6 * 1, hidden_dim=6, output_dim=1, n_heads=2))

    # print("GAT2:")
    # train(GATNet2(input_dim=1, hidden_dim=6, output_dim=1, n_heads=1, T=12))
