import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from graph import get_adjacent_matrix
from torch.utils.data import DataLoader
import os

def get_flow_data(flow_file: str) -> np.array:   # 这个是载入流量数据,返回numpy的多维数组
    """
    :param flow_file: str, path of .npz file to save the traffic flow data
    :return:
        np.array(N, T, D)
    """
    file = os.path.basename(flow_file).split('.')[1]
    if flow_file == '../data/PeMS_04/PeMS04.npz':
        data = np.load(flow_file, allow_pickle=True)

        flow_data = data['data'].transpose([1, 0, 2])[:, :,0 ][:, :, np.newaxis]  # [N, T, D],transpose就是转置，让节点纬度在第0位，N为节点数，T为时间，D为节点特征
        # [:, :, 0]就是只取第一个特征，[:, :, np.newaxis]就是增加一个维度，因为：一般特征比一个多，即使是一个，保持这样的习惯，便于通用的处理问题
        #flow_data = data['data'].transpose([1, 0, 2])[:, :, :]
    elif file =='csv' :
        #csv_reader = csv.reader(flow_file)
        data = []
        for line in csv.reader(open(flow_file), quoting=csv.QUOTE_NONNUMERIC):
            data.append(line)
        # data_nd = np.array(data)
        # print(type(data_nd))
        data = np.array(data)
        flow_data = data[1:][:, np.newaxis].transpose([2, 0, 1])
    else:
        data = np.load(flow_file)

        flow_data = data['train_x'][:, :, 0, :].transpose([1, 0, 2])   #[B, N, T, D],transpose就是转置，让节点纬度在第0位，N为节点数，T为时间，D为节点特征
        #print(flow_data)
        

    return flow_data  # [N, T, D]

class LoadData(Dataset):
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, train_mode):
    # def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, n_pred, train_mode):

        """
        :param data_path: list, ["graph file name", "flow data file name"], path to save the data file names.
        :param num_nodes: int, number of nodes
        :param divide_days: list, [days of train data, days of test data], list to divide to original data
        :param time_interval: int, time interval between two traffic data records(mins)
        :param history_length: int , length of history data to be used
        :param train_mode: list, ["train", "test"]
        """
        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]  # 59-14 = 45 train_data
        self.valid_days = divide_days[1]   
        self.test_days = divide_days[2] # 7*2 = 14 test_data
        self.history_length = history_length  # 30/5 = 6
        #self.n_pred = n_pred
        self.time_interval = time_interval  # 5 min

        self.one_day_length = int(24*60/self.time_interval)
        if data_path[0] =='../data/PeMS_04/PeMS04.csv':
            self.graph = get_adjacent_matrix(distance_file = data_path[0], num_nodes = num_nodes)
        else:
            #csv_reader = csv.reader(open(data_path[0]))
            data = []
            for line in csv.reader(open(data_path[0]), quoting=csv.QUOTE_NONNUMERIC):
                data.append(line)
            # data_nd = np.array(data)
            # print(type(data_nd))
            self.graph = np.array(data)
        self.flow_norm, self.flow_data = self.pre_process_data(data = get_flow_data(data_path[1]), norm_dim =1)

    def __len__(self):  # size of dataset
        """
        :return: length of dataset (number of samples)
        """
        if self.train_mode == "train":
            # return self.train_days * self.one_day_length - self.history_length - self.n_pred # size of train dataset = train - history
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == "valid":
            return self.valid_days * self.one_day_length 
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length  # test = test
        else:
            raise ValueError("Train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # get sample (x, y), index = [0, L1-1]   
        """
        :param item: int, range between [ 0, len-1]
        :return:
            graph: torch.tensor,[N,N]
            data_x: torch.tensor, [N, H, D]
            data_y: torch.tensor, [N, 1, D]
        """
        if self.train_mode =="train":
            index = index
            #print("train index:", index)
        elif self.train_mode == "valid":
            index += self.train_days * self.one_day_length
            #print("valid index:", index)
        elif self.train_mode == "test":
            index += (self.valid_days + self.train_days) * self.one_day_length
            #print("test index:", index)
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))
        data_x, data_y = LoadData.slice_Data(self.flow_data, self.history_length,  index, self.train_mode)
        # data_x, data_y = LoadData.slice_Data(self.flow_data, self.history_length, self.n_pred, index, self.train_mode)
        data_x = LoadData.to_tensor(data_x)  #(N, H, D)
        # data_y = LoadData.to_tensor(data_y) #(N, 1, D)
        data_y = LoadData.to_tensor(data_y).unsqueeze(1) #(N, 1, D)
        return {"graph": LoadData.to_tensor(self.graph), "flow_x":data_x, "flow_y":data_y}
    @staticmethod
    def slice_Data(data, history_length, index, train_mode): #devide the size of dataset based on the history

        """
        :param data: np.array, normalized traffic data
        :param history_length: int, length of history dat tobe used
        :param index: int, index on temporal axis
        :param train_mode: str, ["train", "test"]
        :return:
            data_X: np.array, [N, H, D]
            data_y: np.array, [N, D]
        """
        if train_mode =="train":
            start_index = index
            end_index = index + history_length
            # y_index = end_index + n_pred
        elif train_mode == "valid":
            # start_index = index - history_length - n_pred
            #end_index = index
            start_index = index - history_length
            end_index = index
            # end_index = index - n_pred
            # y_index = end_index + n_pred
        elif train_mode == "test":
            start_index = index- history_length
            end_index = index
            # start_index = index - history_length
            # start_index = index - history_length - n_pred
            # end_index = index - n_pred
            # y_index = end_index + n_pred
            
        else:
            raise ValueError("train mode: [{}] is not defined".format(train_mode))
        data_x = data[:, start_index: end_index]
        # data_y = data[:, end_index: y_index]
        data_y = data[:, end_index]

        return data_x, data_y
    @staticmethod
    def pre_process_data(data, norm_dim):   # normanized data
        """
        :param data: np.array
        :param norm_dim: int, normalized, dim = 1
        :return:
            norm_base:  list, [max_Data, min_data]
            norm_data: np.array, normalized data
        """
        # print(len(data)) 
        norm_base = LoadData.normalize_base(data, norm_dim)
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)
        # print(norm_base, norm_data)

        return norm_base, norm_data

    @staticmethod
    def normalize_base(data, norm_dim):  #normlized base
        """
        :param data: np.array
        :param norm_dim: int, normalization dimension
        :return:
            max_data: np.array
            min_data: np.array
        """
        #print(len(data[0]))
        max_data = np.max(data, norm_dim, keepdims = True)   #[N, T, D], norm_dim = 1, [N, 1, D], keepdims = True
        min_data = np.min(data, norm_dim, keepdims = True)
        #print(max_data, min_data)

        return max_data, min_data

    @staticmethod
    def normalize_data(max_data, min_data, data):  #max-min data
        """
        :param max_data: np.array, max data
        :param min_data: np.array, min data
        :param data: np.array, original traffic data without normalization
        :return:
            np.array, normalized
        """
        mid = min_data
        base = max_data - min_data
        normalized_data = (data-mid) / base

        return normalized_data

    @staticmethod
    def recover_Data(max_data, min_data, data):  #visualization
        """
        :param max_data:  np.array, max data
        :param min_data: np.array, min data
        :param data: np.array, normalized data
        :return:
            recovered_Data: np.array, recovered data
        """
        mid = min_data
        base = max_data - min_data
        #print(len(base))
        #print(len(data))

        recovered_data = data * base + mid

        return recovered_data

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype = torch.float)


    @staticmethod
    def search_data(sequence_length, num_of_depend, label_start_idx,
                    num_for_predict, units, points_per_hour):
        '''
        Parameters
        ----------
        sequence_length: int, length of all history data
        num_of_depend: int,
        label_start_idx: int, the first index of predicting target
        num_for_predict: int, the number of points will be predicted for each sample
        units: int, week: 7 * 24, day: 24, recent(hour): 1
        points_per_hour: int, number of points per hour, depends on data
        Returns
        ----------
        list[(start_idx, end_idx)]
        '''

        if points_per_hour < 0:
            raise ValueError("points_per_hour should be greater than 0!")

        if label_start_idx + num_for_predict > sequence_length:
            return None

        x_idx = []
        for i in range(1, num_of_depend + 1):
            start_idx = label_start_idx - points_per_hour * units * i
            end_idx = start_idx + num_for_predict
            if start_idx >= 0:
                x_idx.append((start_idx, end_idx))
            else:
                return None

        if len(x_idx) != num_of_depend:
            return None

        return x_idx[::-1]

    @staticmethod
    def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                           label_start_idx, num_for_predict, points_per_hour=12):
        '''
        Parameters
        ----------
        data_sequence: np.ndarray
                       shape is (sequence_length, num_of_vertices, num_of_features)
        num_of_weeks, num_of_days, num_of_hours: int
        label_start_idx: int, the first index of predicting target, 预测值开始的那个点
        num_for_predict: int,
                         the number of points will be predicted for each sample
        points_per_hour: int, default 12, number of points per hour
        Returns
        ----------
        week_sample: np.ndarray
                     shape is (num_of_weeks * points_per_hour,
                               num_of_vertices, num_of_features)
        day_sample: np.ndarray
                     shape is (num_of_days * points_per_hour,
                               num_of_vertices, num_of_features)
        hour_sample: np.ndarray
                     shape is (num_of_hours * points_per_hour,
                               num_of_vertices, num_of_features)
        target: np.ndarray
                shape is (num_for_predict, num_of_vertices, num_of_features)
        '''
        week_sample, day_sample, hour_sample = None, None, None

        if label_start_idx + num_for_predict > data_sequence.shape[0]:
            return week_sample, day_sample, hour_sample, None

        if num_of_weeks > 0:
            week_indices = LoadData.search_data(data_sequence.shape[0], num_of_weeks,
                                       label_start_idx, num_for_predict,
                                       7 * 24, points_per_hour)
            if not week_indices:
                return None, None, None, None

            week_sample = np.concatenate([data_sequence[i: j]
                                          for i, j in week_indices], axis=0)

        if num_of_days > 0:
            day_indices = LoadData.search_data(data_sequence.shape[0], num_of_days,
                                      label_start_idx, num_for_predict,
                                      24, points_per_hour)
            if not day_indices:
                return None, None, None, None

            day_sample = np.concatenate([data_sequence[i: j]
                                         for i, j in day_indices], axis=0)

        if num_of_hours > 0:
            hour_indices = LoadData.search_data(data_sequence.shape[0], num_of_hours,
                                       label_start_idx, num_for_predict,
                                       1, points_per_hour)
            if not hour_indices:
                return None, None, None, None

            hour_sample = np.concatenate([data_sequence[i: j]
                                          for i, j in hour_indices], axis=0)

        target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

        return week_sample, day_sample, hour_sample, target
    @staticmethod
    def read_and_generate_dataset(graph_signal_matrix_filename,
                                  num_of_weeks, num_of_days,
                                  num_of_hours, num_for_predict,
                                  points_per_hour=12, save=False):
        '''
        Parameters
        ----------
        graph_signal_matrix_filename: str, path of graph signal matrix file
        num_of_weeks, num_of_days, num_of_hours: int
        num_for_predict: int
        points_per_hour: int, default 12, depends on data

        Returns
        ----------
        feature: np.ndarray,
                 shape is (num_of_samples, num_of_depend * points_per_hour,
                           num_of_vertices, num_of_features)
        target: np.ndarray,
                shape is (num_of_samples, num_of_vertices, num_for_predict)
        '''
        data_seq = np.load(graph_signal_matrix_filename)['data']  # (sequence_length, num_of_vertices, num_of_features)

        all_samples = []
        for idx in range(data_seq.shape[0]):
            sample = LoadData.get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
                continue

            week_sample, day_sample, hour_sample, target = sample

            sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

            if num_of_weeks > 0:
                week_sample = np.expand_dims(week_sample, axis=0)#.transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(week_sample)

            if num_of_days > 0:
                day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(day_sample)

            if num_of_hours > 0:
                hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(hour_sample)

            target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
            sample.append(target)

            time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
            sample.append(time_sample)

            all_samples.append(
                sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

        split_line1 = int(len(all_samples) * 0.6)
        split_line2 = int(len(all_samples) * 0.8)

        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
        validation_set = [np.concatenate(i, axis=0)
                          for i in zip(*all_samples[split_line1: split_line2])]
        testing_set = [np.concatenate(i, axis=0)
                       for i in zip(*all_samples[split_line2:])]

        train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
        val_x = np.concatenate(validation_set[:-2], axis=-1)
        test_x = np.concatenate(testing_set[:-2], axis=-1)

        train_target = training_set[-2]  # (B,N,T)
        val_target = validation_set[-2]
        test_target = testing_set[-2]

        train_timestamp = training_set[-1]  # (B,1)
        val_timestamp = validation_set[-1]
        test_timestamp = testing_set[-1]

        (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

        all_data = {
            'train': {
                'x': train_x_norm,
                'target': train_target,
                'timestamp': train_timestamp,
            },
            'val': {
                'x': val_x_norm,
                'target': val_target,
                'timestamp': val_timestamp,
            },
            'test': {
                'x': test_x_norm,
                'target': test_target,
                'timestamp': test_timestamp,
            },
            'stats': {
                '_mean': stats['_mean'],
                '_std': stats['_std'],
            }
        }
        print('train x:', all_data['train']['x'].shape)
        print('train target:', all_data['train']['target'].shape)
        print('train timestamp:', all_data['train']['timestamp'].shape)
        print()
        print('val x:', all_data['val']['x'].shape)
        print('val target:', all_data['val']['target'].shape)
        print('val timestamp:', all_data['val']['timestamp'].shape)
        print()
        print('test x:', all_data['test']['x'].shape)
        print('test target:', all_data['test']['target'].shape)
        print('test timestamp:', all_data['test']['timestamp'].shape)
        print()
        print('train data _mean :', stats['_mean'].shape, stats['_mean'])
        print('train data _std :', stats['_std'].shape, stats['_std'])

        if save:
            file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
            dirpath = os.path.dirname(graph_signal_matrix_filename)
            filename = os.path.join(dirpath, file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(
                num_of_weeks)) + '_astcgn'
            print('save file:', filename)
            np.savez_compressed(filename,
                                train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                                train_timestamp=all_data['train']['timestamp'],
                                val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                                val_timestamp=all_data['val']['timestamp'],
                                test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                                test_timestamp=all_data['test']['timestamp'],
                                mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                                )
        return all_data

    @staticmethod
    def normalization(train, val, test):
        '''
        Parameters
        ----------
        train, val, test: np.ndarray

        Returns
        ----------
        stats: dict, two keys: mean and std

        train_norm, val_norm, test_norm: np.ndarray,
                                         shape is the same as original

        '''

        assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True)

        def normalize(x):
            return (x - mean) / std

        train_norm = normalize(train)
        val_norm = normalize(val)
        test_norm = normalize(test)

        return {'mean': mean, 'std': std}, train_norm, val_norm, test_norm

    @staticmethod
    def read_and_generate_dataset(graph_signal_matrix_filename,
                                  num_of_weeks, num_of_days,
                                  num_of_hours, num_for_predict,
                                  points_per_hour=12, merge=False):
        '''
        Parameters
        ----------
        graph_signal_matrix_filename: str, path of graph signal matrix file

        num_of_weeks, num_of_days, num_of_hours: int

        num_for_predict: int

        points_per_hour: int, default 12, depends on data

        merge: boolean, default False,
               whether to merge training set and validation set to train model

        Returns
        ----------
        feature: np.ndarray,
                 shape is (num_of_samples, num_of_batches * points_per_hour,
                           num_of_vertices, num_of_features)

        target: np.ndarray,
                shape is (num_of_samples, num_of_vertices, num_for_predict)

        '''
        data_seq = np.load(graph_signal_matrix_filename)['data']

        all_samples = []
        for idx in range(data_seq.shape[0]):
            sample = LoadData.get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if not sample:
                continue

            week_sample, day_sample, hour_sample, target = sample
            all_samples.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 1, 3),
                                    np.expand_dims(day_sample, axis=0).transpose((0, 2, 1, 3)),
                                    np.expand_dims(hour_sample, axis=0).transpose((0, 2, 1, 3)),
                                    np.expand_dims(target, axis=0).transpose((0, 2, 1, 3))[:, :, :, 0])
            ))

        split_line1 = int(len(all_samples) * 0.6)
        split_line2 = int(len(all_samples) * 0.8)

        if not merge:
            training_set = [np.concatenate(i, axis=0)
                            for i in zip(*all_samples[:split_line1])]
        else:
            print('Merge training set and validation set!')
            training_set = [np.concatenate(i, axis=0)
                            for i in zip(*all_samples[:split_line2])]

        validation_set = [np.concatenate(i, axis=0)
                          for i in zip(*all_samples[split_line1: split_line2])]
        testing_set = [np.concatenate(i, axis=0)
                       for i in zip(*all_samples[split_line2:])]

        train_week, train_day, train_hour, train_target = training_set
        val_week, val_day, val_hour, val_target = validation_set
        test_week, test_day, test_hour, test_target = testing_set

        print('training data: week: {}, day: {}, recent: {}, target: {}'.format(
            train_week.shape, train_day.shape,
            train_hour.shape, train_target.shape))
        print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(
            val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
        print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
            test_week.shape, test_day.shape, test_hour.shape, test_target.shape))

        (week_stats, train_week_norm,
         val_week_norm, test_week_norm) = normalization(train_week,
                                                        val_week,
                                                        test_week)

        (day_stats, train_day_norm,
         val_day_norm, test_day_norm) = normalization(train_day,
                                                      val_day,
                                                      test_day)

        (recent_stats, train_recent_norm,
         val_recent_norm, test_recent_norm) = normalization(train_hour,
                                                            val_hour,
                                                            test_hour)

        all_data = {
            'train': {
                'week': train_week_norm,
                'day': train_day_norm,
                'recent': train_recent_norm,
                'target': train_target,
            },
            'val': {
                'week': val_week_norm,
                'day': val_day_norm,
                'recent': val_recent_norm,
                'target': val_target
            },
            'test': {
                'week': test_week_norm,
                'day': test_day_norm,
                'recent': test_recent_norm,
                'target': test_target
            },
            'stats': {
                'week': week_stats,
                'day': day_stats,
                'recent': recent_stats
            }
        }

        return all_data



from torch.autograd import Variable

# if __name__ == '__main__':
#     train_Data = LoadData(data_path = ["data/los_adj.csv", "data/los_speed.csv"], num_nodes = 307, divide_days =[45, 14],
#                           time_interval = 5, history_length=6, train_mode = "train")
#     train_loader = DataLoader(train_Data, batch_size = 64, shuffle = False, num_workers = 8)
#     # train_Data = LoadData(data_path = ["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes = 307, divide_days =[45, 14],
#     #                       time_interval = 5, history_length=6, train_mode = "train")
#     # train_loader = DataLoader(train_Data, batch_size = 64, shuffle = False, num_workers = 8)
#     for _, data in enumerate(train_loader):
#         print(data)
# 
# 
# 

    # print(train_data[0]["flow_x"].size())
    # print(train_data[0]["flow_y"].size())



