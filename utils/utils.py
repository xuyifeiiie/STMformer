import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
import copy
from bisect import bisect
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from data.aggregate_preprocess_data import scan_category
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_string(log, string):
    """打印log"""
    log.write(string + '\n')
    log.flush()
    print(string)


def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    """Disable cudnn to maximize reproducibility 禁用cudnn以最大限度地提高再现性"""
    torch.cuda.cudnn_enabled = False
    """
    cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用
    如果设置为torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法
    然后再设置：torch.backends.cudnn.benchmark = True，当这个flag为True时，将会让程序在开始时花费一点额外时间，
    为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    但由于其是使用非确定性算法，这会让网络每次前馈结果略有差异,如果想要避免这种结果波动，可以将下面的flag设置为True
    """
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_adjacency_matrix(edge_filename, span_df, num_of_instances, type_='connectivity', id_filename=None):
    """
    :param edge_filename: str, csv边信息文件路径
    :param num_of_instances:int, 节点数量
    :param type_:str, {connectivity, distance}
    :param id_filename:str 节点id：索引
    """
    A = np.zeros((int(num_of_instances), int(num_of_instances)), dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(index): node_id for node_id, index in enumerate(f.read().strip().split('\n'))}  # 建立映射列表
        df = pd.read_csv(edge_filename)
        for row in df.values:
            if len(row) != 3:
                continue
            i, j = int(row[0]), int(row[1])
            A[id_dict[i], id_dict[j]] = 1
            A[id_dict[j], id_dict[i]] = 1

        return A

    if edge_filename == "":
        df = span_df
    else:
        df = pd.read_csv(edge_filename)
    for row in df.values:
        if len(row) != 6:
            continue
        i, j, latency, counts = int(row[2]), int(row[3]), float(row[4]), float(row[5])
        if type_ == 'connectivity':
            A[i, j] = 1
            # A[j, i] = 1
        elif type_ == 'counts':
            A[i, j] = counts
            # A[j, i] = 1 / counts
        elif type_ == 'count_div_latency':
            if latency == 0:
                A[i, j] = 0
            else:
                A[i, j] = counts / (latency / 1000000)
            # A[j, i] = 1 / counts
        else:
            raise ValueError("type_ error, must be "
                             "connectivity or distance!")

    return A


def construct_adj_list(A_list, steps):
    """
    Build a spatial-temporal graph
    A: np.ndarray, adjacency matrix, shape is (N, N)
    steps: select a few time steps to build the graph
    return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    """

    N = len(A_list[0])  # Get the number of rows
    adj = np.zeros((N * steps, N * steps))

    for i in range(steps):
        """The diagonal represents the space map of each time step, which is A"""
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_list[i]

    for i in range(N):
        for k in range(steps - 1):
            """Each node will only connect to itself in adjacent time steps"""
            adj[k * N + i, (k + 1) * N + i] = 1.
            adj[(k + 1) * N + i, k * N + i] = 1.

    return adj


def construct_adj(A):
    """
    Build a spatial-temporal graph from a tensor of adjacency matrices
    A_tensor: torch.Tensor, tensor of adjacency matrices, shape is (B, T, N, N)
    return: new adjacency matrix tensor, shape is (B, N * T, N * T)
    """
    B, T, N, _ = A.size()
    NT = N * T

    adj_tensor = torch.zeros(B, NT, NT, dtype=A.dtype, device=A.device)

    for b in range(B):
        for t in range(T):
            """The diagonal represents the space map of each time step, which is A"""
            adj_tensor[b, t * N: (t + 1) * N, t * N: (t + 1) * N] = A[b, t]

        for i in range(N):
            for k in range(T - 1):
                """Each node will only connect to itself in adjacent time steps"""
                adj_tensor[b, k * N + i, (k + 1) * N + i] = 1.
                adj_tensor[b, (k + 1) * N + i, k * N + i] = 1.

    return adj_tensor


def construct_adj_optimized(A):
    B, T, N, _ = A.size()
    NT = N * T

    # 创建一个大的零张量来存储新的邻接矩阵
    adj_tensor = torch.zeros(B, NT, NT, dtype=A.dtype, device=A.device)

    # 使用高级索引填充对角线部分
    lt_idxs = torch.arange(0, NT // N) * N
    t = torch.Tensor(lt_idxs).unsqueeze(1) + torch.arange(N)
    row = t.unsqueeze(1).repeat(1, N, 1).transpose(1, 2)
    col = t.unsqueeze(1).repeat(1, N, 1)

    # 在扁平化的邻接矩阵中填充值
    adj_tensor[:, row, col] = A

    # 处理相邻时间步的连接
    idx_row = torch.arange(NT - N, device=A.device)
    idx_col = idx_row + N
    adj_tensor[:, idx_row, idx_col] = 1
    adj_tensor[:, idx_col, idx_row] = 1

    return adj_tensor


# class DataLoader(object):
#     def __init__(self, xs, ys, batch_size, pad_with_last_sample=False):
#         """
#         Data loader
#         :param xs: training data
#         :param ys: label data
#         :param batch_size:batch size
#         :param pad_with_last_sample: When the remaining data is not enough,
#         whether to copy the last sample to reach the batch size
#         """
#         self.batch_size = batch_size
#         self.current_ind = 0
#         if pad_with_last_sample:
#             num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
#             x_padding = np.repeat(xs[-1:], num_padding, axis=0)
#             y_padding = np.repeat(ys[-1:], num_padding, axis=0)
#             # xs = np.concatenate([xs, x_padding], axis=0)
#             # ys = np.concatenate([ys, y_padding], axis=0)
#             xs = np.concatenate((xs, x_padding), axis=0)
#             ys = np.concatenate((ys, y_padding), axis=0)
#         # len(xs) xs的第一维度
#         self.size = len(xs)
#         self.num_batch = int(self.size // self.batch_size)
#         self.xs = xs
#         self.ys = ys
#
#     def shuffle(self):
#         """Shuffle Dataset"""
#         permutation = np.random.permutation(self.size)
#         xs, ys = self.xs[permutation], self.ys[permutation]
#         self.xs = xs
#         self.ys = ys
#
#     def get_iterator(self):
#         self.current_ind = 0
#
#         def _wrapper():
#             while self.current_ind < self.num_batch:
#                 start_ind = self.batch_size * self.current_ind
#                 end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
#                 x_i = self.xs[start_ind:end_ind, ...]
#                 y_i = self.ys[start_ind:end_ind, ...]
#                 yield x_i, y_i
#                 self.current_ind += 1
#
#         return _wrapper()


def compute_reshape_mean_std_for_4_dimension_data(data):
    """
    :param data: S*T*N*D   4 dimension data
    :return: step_mean: S*TN*D
             step_std: S*TN*D
    """
    a, b, c, d = data.shape
    # to_float = np.vectorize(float)
    # data = to_float(data)
    # squezze the time step dimension (a,b,c,d)->(-1,d)
    reshape_data = data.reshape(-1, d)
    step_mean = reshape_data.mean(axis=0)
    step_std = reshape_data.std(axis=0)
    return step_mean, step_std


class MinMaxNorm:
    def __init__(self, d_min, d_max):
        self.d_min = d_min
        self.d_max = d_max
        eps = 1e-8
        self.denom = self.d_max - self.d_min
        for i in range(len(self.denom)):
            if self.denom[i] == 0:
                self.denom[i] += eps

    def transform(self, data):
        if len(data.shape) == 4:
            a, b, c, d = data.shape
            reshape_data = data.reshape(a, -1, d)
            tmp_norm_data = (reshape_data - self.d_min) / self.denom
            norm_data = tmp_norm_data.reshape(a, -1, c, d)
        elif len(data.shape) == 3:
            b, c, d = data.shape
            reshape_data = data.reshape(-1, d)
            tmp_norm_data = (reshape_data - self.d_min) / self.denom
            norm_data = tmp_norm_data.reshape(-1, c, d)
        return norm_data

    def inverse_transform(self, data):
        if len(data.shape) == 4:
            a, b, c, d = data.shape
            reshape_data = data.reshape(a, -1, d).cpu().detach().numpy()
            tmp_inv_norm_data = (reshape_data * self.denom) + self.d_min
            inverse_data = tmp_inv_norm_data.reshape(a, -1, c, d)
        elif len(data.shape) == 3:
            b, c, d = data.shape
            reshape_data = data.reshape(-1, d).cpu().detach().numpy()
            tmp_inv_norm_data = (reshape_data * self.denom) + self.d_min
            inverse_data = tmp_inv_norm_data.reshape(-1, c, d)
        return inverse_data
class StandardScaler:
    """Standardize the input using standard mean sub and std div"""

    def __init__(self, mean, std, fill_zeros=False):
        self.step_mean = mean
        self.step_std = std
        self.fill_zeros = fill_zeros

    def transform(self, data):
        if len(data.shape) == 4:
            a, b, c, d = data.shape
            # squezze the time step dimension (a, b, c, d)->(a, -1, d)
            reshape_data = data.reshape(a, -1, d)
            # step_mean = np.expand_dims(self.step_mean, axis=1).repeat(b*c, axis=1)
            # step_std = np.expand_dims(self.step_std, axis=1).repeat(b*c, axis=1)
            # tmp_norm_data = reshape_data - step_mean / (step_std + 1e-5)
            tmp_norm_data = (reshape_data - self.step_mean) / (self.step_std + 1e-5)
            norm_data = tmp_norm_data.reshape(a, -1, c, d)
        elif len(data.shape) == 3:
            b, c, d = data.shape
            reshape_data = data.reshape(-1, d)
            tmp_norm_data = (reshape_data - self.step_mean) / (self.step_std + 1e-5)
            norm_data = tmp_norm_data.reshape(-1, c, d)
        return norm_data

    def inverse_transform(self, data):
        a, b, c, d = data.shape
        reshape_data = data.reshape(a, -1, d).cpu().detach().numpy()
        inverse_data = (reshape_data * self.step_std) + self.step_mean
        return inverse_data


class MicroDataset(Dataset):
    def __init__(self, x, y):
        self.len = x.shape[0]  # 获取到数据的总条数（即行）
        self.x_data = torch.Tensor(x)
        self.y_data = torch.Tensor(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def load_dataset(dataset_dir, sample_rate, if_scale=True, if_test=False):
    data = {}
    dataset = {}
    sample_rate_str = str(sample_rate)
    if not if_test:
        for category in ['train', 'val', 'test']:
            print("Loading {} dataset...".format(category))
            cat_data = np.load(os.path.join(dataset_dir, category + '_{}.npz'.format(sample_rate_str)),
                               allow_pickle=True, mmap_mode='r')
            data['x_' + category] = np.array(cat_data['x'][..., :], dtype=float)
            data['y_' + category] = np.array(cat_data['y'][..., :], dtype=float)
            data['adj_in_' + category] = np.array(cat_data['adj_in'][..., :], dtype=float)
            data['adj_out_' + category] = np.array(cat_data['adj_out'][..., :], dtype=float)
            data['l1_' + category] = np.array(cat_data['l1'][..., :], dtype=float)
            data['l2_' + category] = np.array(cat_data['l2'][..., :], dtype=float)
            data['l3_' + category] = np.array(cat_data['l3'][..., :], dtype=float)
            data['instances_count_' + category] = np.array(cat_data['instances_count'][..., :], dtype=float)
    else:
        for category in ['train', 'val', 'test']:
            print("Loading {} dataset...".format(category))
            cat_data = np.load(os.path.join(dataset_dir, category + '_{}.npz'.format(sample_rate_str)),
                               allow_pickle=True)
            data['x_' + category] = np.array(cat_data['x'][:30, ..., :], dtype=float)
            data['y_' + category] = np.array(cat_data['y'][:30, ..., :], dtype=float)
            data['adj_in_' + category] = np.array(cat_data['adj_in'][:30, ..., :], dtype=float)
            data['adj_out_' + category] = np.array(cat_data['adj_out'][:30, ..., :], dtype=float)
            data['l1_' + category] = np.array(cat_data['l1'][:30, ..., :], dtype=float)
            data['l2_' + category] = np.array(cat_data['l2'][:30, ..., :], dtype=float)
            data['l3_' + category] = np.array(cat_data['l3'][:30, ..., :], dtype=float)
            data['instances_count_' + category] = np.array(cat_data['instances_count'][:30, ..., :], dtype=float)

    x_all = np.concatenate((data['x_train'], data['x_val'], data['x_test']), axis=0)
    # y_all = np.concatenate((data['y_train'], data['y_val'], data['y_test']), axis=0)

    if if_scale:
        # data['x_train']是4维数据，num_samples*time_stamps*num_nodes*num_features
        x_mean, x_std = compute_reshape_mean_std_for_4_dimension_data(x_all)
        scaler1 = StandardScaler(mean=x_mean, std=x_std)
        for category in ['train', 'val', 'test']:
            data['x_' + category] = scaler1.transform(data['x_' + category])
        #
        # y_mean, y_std = compute_reshape_mean_std_for_4_dimension_data(y_all)
        # scaler2 = StandardScaler(mean=y_mean, std=y_std)
        for category in ['train', 'val', 'test']:
            data['y_' + category] = scaler1.transform(data['y_' + category])

        dataset['x_scaler'] = scaler1
        dataset['y_scaler'] = scaler1

    dataset['fc_train'] = MicroDataset(data['x_train'], data['y_train'])
    dataset['fc_val'] = MicroDataset(data['x_val'], data['y_val'])
    dataset['fc_test'] = MicroDataset(data['x_test'], data['y_test'])
    dataset['adj_train'] = MicroDataset(data['adj_in_train'], data['adj_out_train'])
    dataset['adj_val'] = MicroDataset(data['adj_in_val'], data['adj_out_val'])
    dataset['adj_test'] = MicroDataset(data['adj_in_test'], data['adj_out_test'])
    dataset['label_train'] = MicroDataset(data['l2_train'], data['l3_train'])
    dataset['label_val'] = MicroDataset(data['l2_val'], data['l3_val'])
    dataset['label_test'] = MicroDataset(data['l2_test'], data['l3_test'])
    dataset['instances_count_train'] = MicroDataset(data['instances_count_train'], data['l1_train'])
    dataset['instances_count_val'] = MicroDataset(data['instances_count_val'], data['l1_val'])
    dataset['instances_count_test'] = MicroDataset(data['instances_count_test'], data['l1_test'])

    return dataset


class ReadData:
    def __init__(self, dataset_path, category_list, sample_rate_str, samples_need):
        self.dataset_path = dataset_path
        self.category_list = category_list
        self.sample_rate_str = sample_rate_str
        self.samples_need = samples_need
        self.category_train_path_list = []
        self.category_val_path_list = []
        self.category_test_path_list = []
        for tvt in ['train', 'val', 'test']:
            for category in self.category_list:
                if sys.platform.startswith('win'):
                    category_path = './' + category
                    data_path = self.dataset_path + \
                                os.path.join(category_path, "{}_{}_{}.npz".format(category, tvt, self.sample_rate_str))
                    if tvt == 'train':
                        self.category_train_path_list.append(data_path)
                    elif tvt == 'val':
                        self.category_val_path_list.append(data_path)
                    elif tvt == 'test':
                        self.category_test_path_list.append(data_path)
                elif sys.platform.startswith('linux'):
                    cat_short = category.split('/')[-1]
                    category_path = category
                    data_path = self.dataset_path + \
                                os.path.join(category_path, "{}_{}_{}.npz".format(cat_short, tvt, self.sample_rate_str))
                    if tvt == 'train':
                        self.category_train_path_list.append(data_path)
                    elif tvt == 'val':
                        self.category_val_path_list.append(data_path)
                    elif tvt == 'test':
                        self.category_test_path_list.append(data_path)

        self.train_data_mmaps = [np.load(dp, mmap_mode='r') for dp in self.category_train_path_list]
        self.val_data_mmaps = [np.load(dp, mmap_mode='r') for dp in self.category_val_path_list]
        self.test_data_mmaps = [np.load(dp, mmap_mode='r') for dp in self.category_test_path_list]

        # record indices and shape
        self.category_train_start_indices = [0] * len(self.category_train_path_list)
        self.category_val_start_indices = [0] * len(self.category_val_path_list)
        self.category_test_start_indices = [0] * len(self.category_test_path_list)
        self.train_num_samples = 0
        for index, cat_data in enumerate(self.train_data_mmaps):
            self.category_train_start_indices[index] = self.train_num_samples
            self.train_num_samples += cat_data['x'].shape[0]
        self.val_num_samples = 0
        for index, cat_data in enumerate(self.val_data_mmaps):
            self.category_val_start_indices[index] = self.val_num_samples
            self.val_num_samples += cat_data['x'].shape[0]
        self.test_num_samples = 0
        for index, cat_data in enumerate(self.test_data_mmaps):
            self.category_test_start_indices[index] = self.test_num_samples
            self.test_num_samples += cat_data['x'].shape[0]

        # cut data for program test
        if self.samples_need != -1:
            self.data_train = {}
            self.data_val = {}
            self.data_test = {}
            for tvt in ['train', 'val', 'test']:
                if self.samples_need <= self.train_data_mmaps[0]['x'].shape[0] and \
                        self.samples_need <= self.val_data_mmaps[0]['x'].shape[0] and \
                        self.samples_need <= self.test_data_mmaps[0]['x'].shape[0]:
                    if tvt == 'train':
                        cat_data = self.train_data_mmaps[0]
                        data_dict = self.data_train
                        self.train_num_samples = self.samples_need
                    elif tvt == 'val':
                        cat_data = self.val_data_mmaps[0]
                        data_dict = self.data_val
                        self.val_num_samples = self.samples_need
                    elif tvt == 'test':
                        cat_data = self.test_data_mmaps[0]
                        data_dict = self.data_test
                        self.test_num_samples = self.samples_need
                    data_dict['x'] = np.array(cat_data['x'][:self.samples_need, ..., :], dtype=np.float32)
                    data_dict['y'] = np.array(cat_data['y'][:self.samples_need, ..., :], dtype=np.float32)
                    data_dict['adj_in'] = np.array(cat_data['adj_in'][:self.samples_need, ..., :],
                                                   dtype=np.float32)
                    data_dict['adj_out'] = np.array(cat_data['adj_out'][:self.samples_need, ..., :],
                                                    dtype=np.float32)
                    data_dict['l1'] = np.array(cat_data['l1'][:self.samples_need, ..., :], dtype=np.float32)
                    data_dict['l2'] = np.array(cat_data['l2'][:self.samples_need, ..., :], dtype=np.float32)
                    data_dict['l3'] = np.array(cat_data['l3'][:self.samples_need, ..., :], dtype=np.float32)
                    data_dict['instances_count'] = np.array(cat_data['instances_count'][:self.samples_need, ..., :],
                                                            dtype=np.float32)

        for tvt in ['train', 'val', 'test']:
            if tvt == 'train':
                self.x_train_shape = [self.train_num_samples, *self.train_data_mmaps[0]['x'][0].shape]
                self.y_train_shape = [self.train_num_samples, *self.train_data_mmaps[0]['y'][0].shape]
                self.adj_train_shape = [self.train_num_samples, *self.train_data_mmaps[0]['adj_in'][0].shape]
                self.label_train_shape = [self.train_num_samples, *self.train_data_mmaps[0]['l2'][0].shape]
                self.instances_count_train_shape = [self.train_num_samples,
                                                    *self.train_data_mmaps[0]['instances_count'][0].shape]
            elif tvt == 'val':
                self.x_val_shape = [self.val_num_samples, *self.val_data_mmaps[0]['x'][0].shape]
                self.y_val_shape = [self.val_num_samples, *self.val_data_mmaps[0]['y'][0].shape]
                self.adj_val_shape = [self.val_num_samples, *self.val_data_mmaps[0]['adj_in'][0].shape]
                self.label_val_shape = [self.val_num_samples, *self.val_data_mmaps[0]['l2'][0].shape]
                self.instances_count_val_shape = [self.val_num_samples,
                                                  *self.val_data_mmaps[0]['instances_count'][0].shape]
            elif tvt == 'test':
                self.x_test_shape = [self.test_num_samples, *self.test_data_mmaps[0]['x'][0].shape]
                self.y_test_shape = [self.test_num_samples, *self.test_data_mmaps[0]['y'][0].shape]
                self.adj_test_shape = [self.test_num_samples, *self.test_data_mmaps[0]['adj_in'][0].shape]
                self.label_test_shape = [self.test_num_samples, *self.test_data_mmaps[0]['l2'][0].shape]
                self.instances_count_test_shape = [self.test_num_samples,
                                                   *self.test_data_mmaps[0]['instances_count'][0].shape]


class LargeScaleDataset(Dataset):
    def __init__(self, data, tvt, data_type):
        #  dataset_path, category_list, sample_rate_str, samples_need
        self.data = data
        self.tvt = tvt
        self.data_type = data_type
        self.data_dict = {}
        # self.dataset_path = dataset_path
        # self.category_list = category_list
        # self.sample_rate_str = sample_rate_str
        # self.samples_need = samples_need

    def __getitem__(self, index):
        if self.data.samples_need != -1:
            if self.tvt == 'train':
                self.data_dict = self.data.data_train
            elif self.tvt == 'val':
                self.data_dict = self.data.data_val
            elif self.tvt == 'test':
                self.data_dict = self.data.data_test

            if self.data_type == 'fc':
                data_x = np.array(self.data_dict['x'][index], dtype=np.float32)
                data_y = np.array(self.data_dict['y'][index], dtype=np.float32)
                return data_x, data_y
            elif self.data_type == 'adj':
                data_adj_in = np.array(self.data_dict['adj_in'][index], dtype=np.float32)
                data_adj_out = np.array(self.data_dict['adj_out'][index], dtype=np.float32)
                return data_adj_in, data_adj_out
            elif self.data_type == 'label':
                data_l2 = np.array(self.data_dict['l2'][index], dtype=np.float32)
                data_l3 = np.array(self.data_dict['l3'][index], dtype=np.float32)
                return data_l2, data_l3
            elif self.data_type == 'instances_count':
                data_instances_count = np.array(self.data_dict['instances_count'][index], dtype=np.float32)
                data_l1 = np.array(self.data_dict['l1'][index], dtype=np.float32)
                return data_instances_count, data_l1
        else:
            category_start_indices = []
            # data_mmaps = []
            if self.tvt == 'train':
                category_start_indices = self.data.category_train_start_indices
                # data_mmaps = self.data.train_data_mmaps
                category_path_list = self.data.category_train_path_list
            elif self.tvt == 'val':
                category_start_indices = self.data.category_val_start_indices
                # data_mmaps = self.data.val_data_mmaps
                category_path_list = self.data.category_val_path_list
            elif self.tvt == 'test':
                category_start_indices = self.data.category_test_start_indices
                # data_mmaps = self.data.test_data_mmaps
                category_path_list = self.data.category_test_path_list
            global_index = bisect(category_start_indices, index) - 1
            local_index = index - category_start_indices[global_index]
            data_mmap = np.load(category_path_list[global_index], mmap_mode='r')

            if self.data_type == 'fc':
                data_x = np.array(data_mmap['x'][local_index], dtype=np.float32)
                data_y = np.array(data_mmap['y'][local_index], dtype=np.float32)
                return data_x, data_y
            elif self.data_type == 'adj':
                data_adj_in = np.array(data_mmap['adj_in'][local_index], dtype=np.float32)
                data_adj_out = np.array(data_mmap['adj_out'][local_index], dtype=np.float32)
                return data_adj_in, data_adj_out
            elif self.data_type == 'label':
                data_l2 = np.array(data_mmap['l2'][local_index], dtype=np.float32)
                data_l3 = np.array(data_mmap['l3'][local_index], dtype=np.float32)
                return data_l2, data_l3
            elif self.data_type == 'instances_count':
                data_instances_count = np.array(data_mmap['instances_count'][local_index], dtype=np.float32)
                data_l1 = np.array(data_mmap['l1'][local_index], dtype=np.float32)
                return data_instances_count, data_l1

    def __len__(self):
        if self.tvt == 'train':
            self.len = self.data.train_num_samples
        elif self.tvt == 'val':
            self.len = self.data.val_num_samples
        elif self.tvt == 'test':
            self.len = self.data.test_num_samples
        return self.len


class IntegratedDataset(Dataset):
    def __init__(self, data, tvt):
        self.data = data
        self.tvt = tvt
        self.data_dict = {}

    def __getitem__(self, index):
        if self.data.samples_need != -1:
            if self.tvt == 'train':
                self.data_dict = self.data.data_train
            elif self.tvt == 'val':
                self.data_dict = self.data.data_val
            elif self.tvt == 'test':
                self.data_dict = self.data.data_test

            for data_type in ['fc', 'adj', 'label', 'instances_count']:
                if data_type == 'fc':
                    data_x = np.array(self.data_dict['x'][index], dtype=np.float32)
                    data_y = np.array(self.data_dict['y'][index], dtype=np.float32)
                elif data_type == 'adj':
                    data_adj_in = np.array(self.data_dict['adj_in'][index], dtype=np.float32)
                    data_adj_out = np.array(self.data_dict['adj_out'][index], dtype=np.float32)
                elif data_type == 'label':
                    data_l2 = np.array(self.data_dict['l2'][index], dtype=np.float32)
                    data_l3 = np.array(self.data_dict['l3'][index], dtype=np.float32)
                elif data_type == 'instances_count':
                    data_instances_count = np.array(self.data_dict['instances_count'][index], dtype=np.float32)
                    data_l1 = np.array(self.data_dict['l1'][index], dtype=np.float32)
            return data_x, data_y, data_adj_in, data_adj_out, data_l2, data_l3, data_instances_count, data_l1
        else:
            category_start_indices = []
            if self.tvt == 'train':
                category_start_indices = self.data.category_train_start_indices
                category_path_list = self.data.category_train_path_list
            elif self.tvt == 'val':
                category_start_indices = self.data.category_val_start_indices
                category_path_list = self.data.category_val_path_list
            elif self.tvt == 'test':
                category_start_indices = self.data.category_test_start_indices
                category_path_list = self.data.category_test_path_list
            global_index = bisect(category_start_indices, index) - 1
            local_index = index - category_start_indices[global_index]
            data_mmap = np.load(category_path_list[global_index], mmap_mode='r')

            for data_type in ['fc', 'adj', 'label', 'instances_count']:
                if data_type == 'fc':
                    data_x = np.array(data_mmap['x'][local_index], dtype=np.float32)
                    data_y = np.array(data_mmap['y'][local_index], dtype=np.float32)
                elif data_type == 'adj':
                    data_adj_in = np.array(data_mmap['adj_in'][local_index], dtype=np.float32)
                    data_adj_out = np.array(data_mmap['adj_out'][local_index], dtype=np.float32)
                elif data_type == 'label':
                    data_l2 = np.array(data_mmap['l2'][local_index], dtype=np.float32)
                    data_l3 = np.array(data_mmap['l3'][local_index], dtype=np.float32)
                elif data_type == 'instances_count':
                    data_instances_count = np.array(data_mmap['instances_count'][local_index], dtype=np.float32)
                    data_l1 = np.array(data_mmap['l1'][local_index], dtype=np.float32)
            return data_x, data_y, data_adj_in, data_adj_out, data_l2, data_l3, data_instances_count, data_l1

    def __len__(self):
        if self.tvt == 'train':
            self.len = self.data.train_num_samples
        elif self.tvt == 'val':
            self.len = self.data.val_num_samples
        elif self.tvt == 'test':
            self.len = self.data.test_num_samples
        return self.len

def prepare_and_load_dataset(dataset_path, sample_rate, if_scale=True, if_test=False):
    """"
    read from csv
    """
    dataset = {}
    sample_rate_str = str(sample_rate)
    category_list = scan_category(dataset_path)

    print("Beginning to preprocess and load dataset:...")

    if not if_test:
        samples_need = -1
    else:
        samples_need = 50

    if if_scale:
        print("Beginning to read scaler of data...")
        # mean_std_npz_path = dataset_path + 'mean_std.npz'
        scaler_npz_path = dataset_path + 'scaler.npz'
        scaler_data = np.load(scaler_npz_path, mmap_mode='r', allow_pickle=True)
        # scaler = StandardScaler(mean=data['mean'], std=data['std'])
        scaler = MinMaxNorm(d_min=scaler_data['min'], d_max=scaler_data['max'])
        dataset['scaler'] = scaler
    else:
        scaler = None
        print("Skip normalization phase.")

    print("Beginning to load dataset...")
    data = ReadData(dataset_path, category_list, sample_rate_str, samples_need)

    # dataset['fc_train'] = LargeScaleDataset(data, tvt='train', data_type='fc')
    # dataset['fc_val'] = LargeScaleDataset(data, tvt='val', data_type='fc')
    # dataset['fc_test'] = LargeScaleDataset(data, tvt='test', data_type='fc')
    # dataset['adj_train'] = LargeScaleDataset(data, tvt='train', data_type='adj')
    # dataset['adj_val'] = LargeScaleDataset(data, tvt='val', data_type='adj')
    # dataset['adj_test'] = LargeScaleDataset(data, tvt='test', data_type='adj')
    # dataset['label_train'] = LargeScaleDataset(data, tvt='train', data_type='label')
    # dataset['label_val'] = LargeScaleDataset(data, tvt='val', data_type='label')
    # dataset['label_test'] = LargeScaleDataset(data, tvt='test', data_type='label')
    # dataset['instances_count_train'] = LargeScaleDataset(data, tvt='train', data_type='instances_count')
    # dataset['instances_count_val'] = LargeScaleDataset(data, tvt='val', data_type='instances_count')
    # dataset['instances_count_test'] = LargeScaleDataset(data, tvt='test', data_type='instances_count')
    dataset['train'] = IntegratedDataset(data, tvt='train')
    dataset['val'] = IntegratedDataset(data, tvt='val')
    dataset['test'] = IntegratedDataset(data, tvt='test')

    print("Dataset has been preprocessed and loaded!")
    return dataset
