import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
import tqdm
import argparse
import numpy as np
import pandas as pd
from data.aggregate_preprocess_data import scan_category
from utils.utils import StandardScaler, MinMaxNorm


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


def preprocess_df(df):
    # df.drop(["Label1", "Label2", "Label3"], axis=1, inplace=True)
    # df.drop(["TimeStamp", "Time", "Node", "PodName"], axis=1, inplace=True)
    df.drop(["NodeCpuStat_LogicalCores", "NodeNetStat_Up"], axis=1, inplace=True)
    for column in df.columns.tolist():
        if column.startswith('CNet'):
            # ns -> ms
            if column in ['CNetConnectionDuration', 'CNetConnectionLatency', 'CNetPassiveEstablishSpan',
                          'CNetSingleRecvSpanNs', 'CNetSingleSendSpanNs', 'CNetSingleTransLatencyNs',
                          'CNetStateSpanFromClose', 'CNetStateSpanFromSynrecv']:
                df[column] = df[column] / 1000000
            if column == 'CNetRttVar':
                df[column] = df[column] / 200000
            elif column in ['CNetConnectionDuration', 'CNetSingleTransLatencyNs', 'CNetPassiveEstablishSpan',
                            'CNetRecvPackets', 'CNetRecvBytes', 'CNetSentBytes', 'CNetSentPackets',
                            'CNetStateSpanFromClose', 'CNetStateSpanFromSynrecv']:
                df[column] = df[column].apply(np.log1p)
        if column in ['JvmStat_Size', 'JvmStat_Used', 'JvmStat_JvmGCTime', 'JvmStat_JvmSafepointSyncTime',
                      'JvmStat_JvmSafepointTime', 'NodeDiskStat_BytesReadVar', 'NodeDiskStat_BytesWrittenVar',
                      'NodeMemStat_AvailableBytes', 'NodeMemStat_CachedBytes', 'NodeMemStat_FreeBytes',
                      'NodeMemStat_TotalBytes', 'NodeNetStat_RxBytesVar', 'NodeNetStat_RxPacketsVar',
                      'NodeNetStat_TxBytesVar', 'NodeNetStat_TxPacketsVar']:
            df[column] = df[column].apply(np.log1p)
    # return df, df.min(), df.max()
    return df


def update_min_max(old_min, old_max, new_min, new_max):
    assert len(old_min) == len(new_min)
    assert len(old_max) == len(new_max)
    # min
    for i in range(len(old_min)):
        # if i in ["TimeStamp", "Label1", "Label2", "Label3"]:
        #     continue
        if new_min[i] <= old_min[i]:
            old_min[i] = new_min[i]
        else:
            continue
    # max
    for i in range(len(old_max)):
        # if i in ["TimeStamp", "Label1", "Label2", "Label3"]:
        #     continue
        if new_max[i] >= old_max[i]:
            old_max[i] = new_max[i]
        else:
            continue
    return old_min, old_max


def generate_graph_seq2seq_io_data(category_metric_df, category_instance_df, category_connection_df, x_offsets,
                                   y_offsets, adj_in_offsets, adj_out_offsets, sample_rate, num_of_instances):
    """
    Generate samples in the pattern of H history steps data mapping the P predict steps
    In this function, P = H. And P[0] is the next time stamp of H[-1].
    :param category_metric_df:
    :param category_instance_df:
    :param category_connection_df:
    :param x_offsets:
    :param y_offsets:
    :param adj_in_offsets:
    :param adj_out_offsets:
    :param sample_rate:
    :param num_of_instances:

    :return:
    # x: (time_steps, input_length, num_nodes, input_dim)
    # y: (time_steps, output_length, num_nodes, output_dim)

    """

    num_samples, num_features = category_metric_df.shape
    x, y = [], []
    l1, l2, l3 = [], [], []
    adj_in, adj_out = [], []
    adj_list = []

    data = category_metric_df.values

    tmp_data = data[:, 4:-3]
    arr_min = tmp_data.min(axis=0)
    arr_max = tmp_data.max(axis=0)

    time_steps = num_samples / num_of_instances
    data = np.delete(data, (0, 1, 2, 3), axis=1)
    data = data.reshape(int(time_steps), num_of_instances, -1)

    min_t = abs(min(x_offsets))  # -63--0  63
    max_t = abs(int(time_steps) - abs(max(y_offsets)))  # 1--64  130-64 = 66
    for t in range(min_t, max_t):
        x.append(data[(t + x_offsets), :, :-3])
        y.append(data[(t + y_offsets), :, :-3])
        l1.append(data[(t + y_offsets), :, -3:-2])
        l2.append(data[(t + y_offsets), :, -2:-1])
        l3.append(data[(t + y_offsets), :, -1:])

    start_stamp = category_metric_df.iloc[0]["TimeStamp"]
    end_stamp = category_metric_df.iloc[-1]["TimeStamp"]
    tmp_timestamp = start_stamp - int(sample_rate)
    while tmp_timestamp + int(sample_rate) <= end_stamp:
        span_df = category_connection_df[
            (category_connection_df['TimeStamp'] > tmp_timestamp) & (
                    category_connection_df['TimeStamp'] <= tmp_timestamp + int(sample_rate))]
        adj = get_adjacency_matrix("", span_df, num_of_instances, type_="counts")
        adj_list.append(adj)
        tmp_timestamp = tmp_timestamp + int(sample_rate)

    adj_array = np.array(adj_list)

    for t in range(min_t, max_t):
        adj_in.append(adj_array[(t + adj_in_offsets)])
        adj_out.append(adj_array[(t + adj_out_offsets)])

    x_array = np.stack(x, axis=0)
    y_array = np.stack(y, axis=0)
    adj_in_array = np.stack(adj_in, axis=0)
    adj_out_array = np.stack(adj_out, axis=0)
    l1_array = np.stack(l1, axis=0)
    l2_array = np.stack(l2, axis=0)
    l3_array = np.stack(l3, axis=0)

    a, b, c, d = x_array.shape
    instances_count_on_vm = category_instance_df.values
    instances_count_on_vm = np.expand_dims(instances_count_on_vm, axis=0)
    instances_count_on_vm = np.expand_dims(instances_count_on_vm, axis=0)
    instances_count_on_vm_array = instances_count_on_vm.repeat(b, axis=1).repeat(a, axis=0)

    # 设置数据类型
    x_array = x_array.astype('float32')
    y_array = y_array.astype('float32')
    adj_in_array = adj_in_array.astype('float16')
    adj_out_array = adj_out_array.astype('float16')
    l1_array = l1_array.astype('int8')
    l2_array = l2_array.astype('int8')
    l3_array = l3_array.astype('int8')
    instances_count_on_vm_array = instances_count_on_vm_array.astype('int8')
    return x_array, y_array, adj_in_array, adj_out_array, l1_array, l2_array, l3_array, instances_count_on_vm_array, \
        arr_min, arr_max


def shuffle_before_load(x, y, adj_in, adj_out, l1, l2, l3, instances_count):
    """Shuffle Dataset"""
    permutation = np.random.permutation(x.shape[0])
    xs, ys = x[permutation], y[permutation]
    ins, outs = adj_in[permutation], adj_out[permutation]
    l1s, l2s, l3s = l1[permutation], l2[permutation], l3[permutation]
    instances_counts = instances_count[permutation]
    return xs, ys, ins, outs, l1s, l2s, l3s, instances_counts


def generate_train_val_test_for_metric(args, task_dataset_path):
    """
    Generating train val test data for training.
    Reading data from processed data csv file.
    :param task_dataset_path:
    :param args:
    :return:
    """
    if sys.platform.startswith('win'):
        base_path = args.data_path
    elif sys.platform.startswith('linux'):
        base_path = './'
    sample_rate_str = str(args.sample_rate)
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    category_list = scan_category(base_path)
    x_all = []
    y_all = []
    adj_in_all = []
    adj_out_all = []
    l1_all = []
    l2_all = []
    l3_all = []
    instances_count_all = []

    min_final = None
    max_final = None

    scaler_npz_path = args.data_path + 'scaler.npz'
    scaler_data = np.load(scaler_npz_path, mmap_mode='r', allow_pickle=True)
    min_final = scaler_data['min']
    max_final = scaler_data['max']

    if args.if_scale:
        print("Beginning to normalize data...")
        scaler = MinMaxNorm(min_final, max_final)

        for category in tqdm.tqdm(category_list):
            for cat in ["train", "val", "test"]:
                if category.find("-" + sample_rate_str + "s") == -1:
                    continue
                if sys.platform.startswith('win'):
                    cat_short = category.split('\\')[-1]
                    npz_path = args.data_path + os.path.join(category,
                                                             "{}_{}_{}.npz".format(cat_short, cat, sample_rate_str))
                elif sys.platform.startswith('linux'):
                    cat_short = category.split('/')[-1]
                    npz_path = args.data_path + os.path.join(category,
                                                             "{}_{}_{}.npz".format(cat_short, cat, sample_rate_str))
                cat_data = np.load(npz_path, mmap_mode='c')
                data_x = np.array(cat_data['x'][..., :], dtype=float)
                data_y = np.array(cat_data['y'][..., :], dtype=float)
                data_x = scaler.transform(data_x).astype('float32')
                data_y = scaler.transform(data_y).astype('float32')
                # cat_data['adj_in'][..., :] = np.array(cat_data['adj_in'][..., :], dtype=float).astype('float16')
                # cat_data['adj_out'][..., :] = np.array(cat_data['adj_out'][..., :], dtype=float).astype('float16')
                # cat_data['l1'][..., :] = np.array(cat_data['l1'][..., :], dtype=float).astype('int8')
                # cat_data['l2'][..., :] = np.array(cat_data['l2'][..., :], dtype=float).astype('int8')
                # cat_data['l3'][..., :] = np.array(cat_data['l3'][..., :], dtype=float).astype('int8')
                # cat_data['instances_count'][..., :] = np.array(cat_data['instances_count'][..., :], dtype=float).astype('int8')

                # data_x = np.array(cat_data['x'][..., :], dtype=float)
                # data_y = np.array(cat_data['y'][..., :], dtype=float)
                # data_x = scaler.transform(data_x)
                # data_y = scaler.transform(data_y)
                # data_adj_in = np.array(cat_data['adj_in'][..., :], dtype=float)
                # data_adj_out = np.array(cat_data['adj_out'][..., :], dtype=float)
                # data_l1 = np.array(cat_data['l1'][..., :], dtype=float)
                # data_l2 = np.array(cat_data['l2'][..., :], dtype=float)
                # data_l3 = np.array(cat_data['l3'][..., :], dtype=float)
                # data_instances_count = np.array(cat_data['instances_count'][..., :], dtype=float)

                # data_x = data_x.astype('float32')
                # data_y = data_y.astype('float32')
                # data_adj_in = data_adj_in.astype('float16')
                # data_adj_out = data_adj_out.astype('float16')
                # data_l1 = data_l1.astype('int8')
                # data_l2 = data_l2.astype('int8')
                # data_l3 = data_l3.astype('int8')
                # data_instances_count = data_instances_count.astype('int8')

                np.savez(
                    npz_path,
                    x=data_x,
                    y=data_y,
                    adj_in=np.array(cat_data['adj_in'][..., :], dtype=float).astype('float16'),
                    adj_out=np.array(cat_data['adj_out'][..., :], dtype=float).astype('float16'),
                    l1=np.array(cat_data['l1'][..., :], dtype=float).astype('int8'),
                    l2=np.array(cat_data['l2'][..., :], dtype=float).astype('int8'),
                    l3=np.array(cat_data['l3'][..., :], dtype=float).astype('int8'),
                    instances_count=np.array(cat_data['instances_count'][..., :], dtype=float).astype('int8')
                )
            print('{} has been normalized successfully!'.format(cat_short))
            print('*' * 100)

        print("Normalization phase has been completed.")

    # free mem
    # data_x = None
    # data_y = None
    # data_adj_in = None
    # data_adj_out = None
    # data_l1 = None
    # data_l2 = None
    # data_l3 = None
    # data_instances_count = None

    print("-" * 100)
    print("All category dataset has been processed and saved successfully!")

    # print("Beginning to integrate all data...")
    # for tvt in ['train', 'val', 'test']:
    #     for category in category_list:
    #         if category.find("-" + sample_rate_str + "s") == -1:
    #             continue
    #         if sys.platform.startswith('win'):
    #             # cat_short = category.split('\\')[-1]
    #             npz_path = os.path.join(category, "{}_{}_{}.npz".format(category, tvt, sample_rate_str))
    #         elif sys.platform.startswith('linux'):
    #             cat_short = category.split('/')[-1]
    #             npz_path = os.path.join(category, "{}_{}_{}.npz".format(cat_short, tvt, sample_rate_str))
    #         # allow_pickle=True,
    #         cat_data = np.load(npz_path, mmap_mode='r')
    #
    #         x_all.append(cat_data['x'])
    #         y_all.append(cat_data['y'])
    #         adj_in_all.append(cat_data['adj_in'])
    #         adj_out_all.append(cat_data['adj_out'])
    #         l1_all.append(cat_data['l1'])
    #         l2_all.append(cat_data['l2'])
    #         l3_all.append(cat_data['l3'])
    #         instances_count_all.append(cat_data['instances_count'])
    #
    # x_all_array = np.concatenate(x_all, axis=0)
    # x_all.clear()
    # y_all_array = np.concatenate(y_all, axis=0)
    # y_all.clear()
    # adj_in_all_array = np.concatenate(adj_in_all, axis=0)
    # adj_in_all.clear()
    # adj_out_all_array = np.concatenate(adj_out_all, axis=0)
    # adj_out_all.clear()
    # l1_all_array = np.concatenate(l1_all, axis=0)
    # l1_all.clear()
    # l2_all_array = np.concatenate(l2_all, axis=0)
    # l2_all.clear()
    # l3_all_array = np.concatenate(l3_all, axis=0)
    # l3_all.clear()
    # instances_count_all_array = np.concatenate(instances_count_all, axis=0)
    # instances_count_all.clear()
    #
    # if args.if_shuffle:
    #     x_all_array, y_all_array, adj_in_all_array, adj_out_all_array, l1_all_array, l2_all_array, l3_all_array, instances_count_all_array = shuffle_before_load(
    #         x_all_array, y_all_array, adj_in_all_array, adj_out_all_array, l1_all_array, l2_all_array, l3_all_array,
    #         instances_count_all_array)
    #
    # # Write the data into npz file.
    # num_samples = x_all_array.shape[0]
    # num_test = round(num_samples * 0.10)  # 10 % testing
    # num_train = round(num_samples * 0.80)  # 80 % training
    # num_val = num_samples - num_test - num_train  # 10 % validation
    #
    # x_train, y_train = x_all_array[:num_train], y_all_array[:num_train]
    # adj_in_train, adj_out_train = adj_in_all_array[:num_train], adj_out_all_array[:num_train]
    # l1_train, l2_train, l3_train = l1_all_array[:num_train], l2_all_array[:num_train], l3_all_array[:num_train]
    # instances_count_train = instances_count_all_array[:num_train]
    #
    # x_val, y_val = (
    #     x_all_array[num_train: num_train + num_val],
    #     y_all_array[num_train: num_train + num_val],
    # )
    # adj_in_val, adj_out_val = (
    #     adj_in_all_array[num_train: num_train + num_val],
    #     adj_out_all_array[num_train: num_train + num_val],
    # )
    # l1_val, l2_val, l3_val = (
    #     l1_all_array[num_train: num_train + num_val],
    #     l2_all_array[num_train: num_train + num_val],
    #     l3_all_array[num_train: num_train + num_val],
    # )
    # instances_count_val = instances_count_all_array[num_train: num_train + num_val]
    #
    # x_test, y_test = x_all_array[-num_test:], y_all_array[-num_test:]
    # adj_in_test, adj_out_test = adj_in_all_array[-num_test:], adj_out_all_array[-num_test:]
    # l1_test, l2_test, l3_test = l1_all_array[-num_test:], l2_all_array[-num_test:], l3_all_array[-num_test:]
    # instances_count_test = instances_count_all_array[-num_test:]
    #
    # for cat in ["train", "val", "test"]:
    #     # locals() 函数会以字典类型返回当前位置的全部局部变量。
    #     _x, _y = locals()["x_" + cat], locals()["y_" + cat]
    #     _adj_in, _adj_out = locals()["adj_in_" + cat], locals()["adj_out_" + cat]
    #     _l1, _l2, _l3 = locals()["l1_" + cat], locals()["l2_" + cat], locals()["l3_" + cat]
    #     _instances_count = locals()["instances_count_" + cat]
    #     print(cat, "x: ", _x.shape, "y:", _y.shape)
    #     print(cat, "adj_in: ", _adj_in.shape, "adj_out:", _adj_out.shape)
    #     print(cat, "l1: ", _l1.shape, "l2:", _l2.shape, "l3:", _l3.shape)
    #     print(cat, "instances_count: ", _instances_count.shape)
    #
    #     np.savez(
    #         os.path.join(task_dataset_path, "{}_{}.npz".format(cat, sample_rate_str)),
    #         x=_x,
    #         y=_y,
    #         adj_in=_adj_in,
    #         adj_out=_adj_out,
    #         l1=_l1,
    #         l2=_l2,
    #         l3=_l3,
    #         instances_count=_instances_count
    #     )
    # print("All data has been integrated together!")


if __name__ == "__main__":
    """
    task_name: 
        forecast
        anomaly detection
        imputation
        classification

    """
    parser = argparse.ArgumentParser()

    # designed to generate dataset on Windows
    parser.add_argument("--output_dir", type=str, default="./processed",
                        help="Output directory.")
    parser.add_argument("--data_path", type=str, default="./",
                        help="Raw Data.", )
    parser.add_argument("--task_name", type=str, default="short_term_forecast",
                        help="Task name", )
    parser.add_argument("--seq_length_x", type=int, default=128, help="Input Sequence Length.", )
    parser.add_argument("--seq_length_y", type=int, default=128, help="Output Sequence Length.", )
    parser.add_argument("--num_of_instances", type=int, default=56, help="Nodes number", )
    parser.add_argument("--sample_rate", type=int, default=5, help="the sample rate of collecting metrics", )
    parser.add_argument("--if_shuffle", type=eval, default=False, help="Whether to shuffle the dataset", )
    parser.add_argument('--if_scale', type=eval, default=True, help="Whether to scale the raw data")
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--type_of_adj", type=str, default="counts",
                        help="choose the type of adj", )

    args = parser.parse_args()
    task_dataset_path = args.output_dir + '/' + args.task_name
    # if os.path.exists(task_dataset_path):
    #     reply = str(input(f'{task_dataset_path} exists. Do you want to overwrite it? (y/n)')).lower().strip()
    #     if reply[0] != 'y':
    #         sys.exit('Did not overwrite file.')
    # else:
    #     os.makedirs(task_dataset_path)
    if os.path.exists(task_dataset_path):
        if len(os.listdir(task_dataset_path)) == 0:
            os.rmdir(task_dataset_path)
            os.makedirs(task_dataset_path)
    else:
        os.makedirs(task_dataset_path)
    generate_train_val_test_for_metric(args, task_dataset_path)
