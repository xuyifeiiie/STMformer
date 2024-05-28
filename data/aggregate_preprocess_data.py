import ast
import math
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import json
import tqdm


def get_subfiles_subfolders(folder_path):
    """
    Get subfile and subfolders of a directory
    :param folder_path: directory path
    :return: subfile and subfolders path list
    """
    files = []
    subfolders = []
    if sys.platform.startswith('win'):
        abs_path = os.path.abspath(folder_path)
    elif sys.platform.startswith('linux'):
        abs_path = folder_path
    # abs_path = os.path.abspath(folder_path)
    for file in os.listdir(abs_path):
        file_path = os.path.join(abs_path, file)
        if os.path.isdir(file_path):
            subfolders.append(file_path)
        else:
            files.append(file_path)
    return files, subfolders


def scan_category(base_path):
    _, base_folders = get_subfiles_subfolders(base_path)
    category_list = []
    for category_path in base_folders:
        if os.path.isdir(category_path):
            if sys.platform.startswith('win'):
                category = category_path.split("\\")[-1]
                if category.find("ipynb") != -1:
                    continue
                elif category.find("processed") != -1:
                    continue
                elif category.find("pycache") != -1:
                    continue
                else:
                    category_list.append(category)
            elif sys.platform.startswith('linux'):
                category = category_path.split("/")[-1]
                if category.find("ipynb") != -1:
                    continue
                elif category.find("processed") != -1:
                    continue
                elif category.find("pycache") != -1:
                    continue
                else:
                    category_list.append(category)

    return category_list


def set_timestamp(df, start_timestamp, end_timestamp, interval=12):
    """
    used for reset the timestamp of raw data. Some data collector not synchronize steps for
    collecting data, so need to reset the timestamp in the interval into unified stamp.
    :param df:
    :param start_timestamp: timestamp
    :param end_timestamp: timestamp
    :param interval: 10s
    :return:
    """
    # start_timestamp = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp()
    # end_timestamp = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp()
    interval_start_timestamp = start_timestamp
    delete_index = []
    time_delta = end_timestamp - start_timestamp
    if time_delta % interval != 0:
        num_interval = time_delta // interval
        last_end_timestamp = start_timestamp + num_interval * interval
    else:
        last_end_timestamp = end_timestamp
    df.insert(loc=0, column="TimeStamp", value=None)
    while (interval_start_timestamp + interval) <= last_end_timestamp:
        interval_end_timestamp = interval_start_timestamp + interval
        for index, row in df.iterrows():
            row_timestamp = datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S").timestamp()
            if interval_start_timestamp <= row_timestamp < interval_end_timestamp:
                tmp = datetime.fromtimestamp(interval_start_timestamp)
                df.at[index, "TimeStamp"] = interval_start_timestamp
                df.at[index, "Time"] = tmp.strftime("%Y-%m-%d %H:%M:%S")
            elif row_timestamp < start_timestamp or row_timestamp >= last_end_timestamp:
                delete_index.append(index)
            else:
                continue
        interval_start_timestamp = interval_end_timestamp
    if len(delete_index) == 0:
        return df
    else:
        df = df.drop(index=delete_index)
        return df


def read_metric_from_log(file_path):
    """
    Read data from metric log file
    The time and timestamp set by the log record timestamp.
    :param file_path:
    :return: df
    """
    i = 0
    log_content = pd.read_csv(file_path, sep="\t", header=None)
    for index, row in log_content.iterrows():
        data_array = row.values
        data_str = np.array2string(data_array)
        data_str = data_str.replace("['", "'").replace("']", "'").replace('false', 'False') \
            .replace('true', 'True').replace('null', 'None')
        data_str = data_str.replace('null', 'None')
        data_dict = ast.literal_eval(data_str)
        data_dict = ast.literal_eval(data_dict)
        if data_dict["log"]["event"]["monitor_type"] == "ebpf":
            data = data_dict["log"]["event"]["data"]["event_ebpf"]
            data_df = pd.DataFrame(data=data, index=[index])
        elif data_dict["log"]["event"]["monitor_type"] == "metric":
            data = data_dict["log"]["event"]["data"]["metric"]
            data_df = pd.DataFrame(data=data, index=[index])
            data_df["NodeStats"] = json.dumps(data["NodeStats"])
            data_df["CgroupStats"] = json.dumps(data["CgroupStats"])
            data_df["ContainerNet"] = json.dumps(data["ContainerNet"])
            tmp_timestamp = int(data["TimeStamp"] / 1000000000)
            data_df["TimeStamp"] = tmp_timestamp
            tmp = datetime.fromtimestamp(tmp_timestamp)
            data_df["Time"] = tmp.strftime("%Y-%m-%d %H:%M:%S")
        data_df["Node"] = data_dict["log"]["id"]
        i += 1
        if i == 1:
            base_df = data_df
        else:
            cat_df = pd.concat([base_df, data_df])
            base_df = cat_df
    return base_df


def read_trace_from_log(file_path):
    """
    Read trace data from log files.
    The time and timestamp set by the log record time, since the record timestamp is the virtual machine timestamp
    :param file_path:
    :return:
    """
    i = 0
    try:
        log_content = pd.read_csv(file_path, sep="\t", header=None)
        if log_content.empty:
            print("trace file is empty!")
        else:
            print()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    for index, row in log_content.iterrows():
        data_array = row.values
        data_str = np.array2string(data_array)
        data_str = data_str.replace("['", "'").replace("']", "'").replace('false', 'False') \
            .replace('true', 'True').replace('null', 'None')
        data_dict = ast.literal_eval(data_str)
        data_dict = ast.literal_eval(data_dict)
        if data_dict["log"]["event"]["monitor_type"] == "ebpf":
            data = data_dict["log"]["event"]["data"]["event_ebpf"]
            data_df = pd.DataFrame(data=data, index=[index])
            data_df.insert(loc=0, column="Node", value=None)
            data_df.insert(loc=0, column="Time", value=None)
            data_df.insert(loc=0, column="TimeStamp", value=None)
            data_df["Node"] = data_dict["log"]["id"]
            if isinstance(data_dict["time"], str):
                time_tmp = str.split(data_dict["time"], '+')[0]
                time_need = str.replace(time_tmp, "T", " ")
                data_df["Time"] = time_need
                data_df["TimeStamp"] = datetime.strptime(time_need, "%Y-%m-%d %H:%M:%S").timestamp()
        i += 1
        if i == 1:
            base_df = data_df
        else:
            cat_df = pd.concat([base_df, data_df])
            base_df = cat_df
    return base_df


def flatten_metric_data(df, interval):
    ndf = pd.DataFrame(columns=["TimeStamp", "Time", "Node", "ContainerInfo", "Namespace", "PodName", "NodeName",
                                "ContainerName", "ApplicationType", "Restarts", "RestartsVar", "OomKills",
                                "OomKillsVar",
                                "CgroupCpuStat_UsageSeconds", "CgroupCpuStat_UsageSecondsVar",
                                "CgroupCpuStat_ThrottledTimeSeconds", "CgroupCpuStat_ThrottledTimeSecondsVar",
                                "CgroupCpuStat_ThrottledPercent", "CgroupCpuStat_CpuUserUsagePercent",
                                "CgroupCpuStat_CpuSysUsagePercent", "CgroupCpuStat_LimitCores",
                                "CgroupMemStat_RSS", "CgroupMemStat_Cache", "CgroupMemStat_UsedBytes",
                                "CgroupMemStat_UsedPercent", "CgroupMemStat_Limit",
                                "CgroupIOStat_ReadOps", "CgroupIOStat_WriteOps", "CgroupIOStat_ReadBytes",
                                "CgroupIOStat_WrittenBytes", "CgroupIOStat_ReadOpsVar", "CgroupIOStat_WriteOpsVar",
                                "CgroupIOStat_ReadBytesVar", "CgroupIOStat_WrittenBytesVar",
                                "CgroupFsStat_CapacityBytes", "CgroupFsStat_UsedBytes", "CgroupFsStat_ReservedBytes",
                                "CgroupFsStat_CapacityBytesVar", "CgroupFsStat_UsedBytesVar",
                                "CgroupFsStat_ReservedBytesVar", "CgroupDelays_Cpu", "CgroupDelays_CpuVar",
                                "CgroupDelays_Disk", "CgroupDelays_DiskVar",
                                "JvmStat_JvmInfo", "JvmStat_Size", "JvmStat_Used", "JvmStat_JvmGCTime",
                                "JvmStat_JvmSafepointTime", "JvmStat_JvmSafepointSyncTime",
                                "CNetListenopen", "CNetConnectsSuccessful", "CNetConnectsFailed",
                                "CNetRetransmits", "CNetDrops", "CNetConnectionsActive", "CNetListenopenVar",
                                "CNetConnectsSuccessfulVar", "CNetConnectsFailedVar", "CNetRetransmitsVar",
                                "CNetDropsVar", "CNetConnectionsActiveVar", "CNetConnectionLatency",
                                "CNetConnectionDuration", "CNetSrtt", "CNetRttVar", "CNetStateSpanFromClose",
                                "CNetPassiveEstablishSpan", "CNetStateSpanFromSynrecv", "CNetSingleTransLatencyNs",
                                "CNetSingleSendSpanNs", "CNetSingleRecvSpanNs", "CNetSentBytes", "CNetSentBytesAver",
                                "CNetRecvBytes", "CNetRecvBytesAver", "CNetSentPackets", "CNetRecvPackets",
                                "CNetLastSentPackets", "CNetLastRecvPackets",
                                "CNetWriteBufferMaxUsage", "CNetReadBufferMaxUsage", "CNetSendAsClient",
                                "CNetSendAsServer", "CNetRecvAsClient", "CNetRecvAsServer", "CNetSendAsClientVar",
                                "CNetSendAsServerVar", "CNetRecvAsClientVar", "CNetRecvAsServerVar",
                                "NodeCpuStat_TotalUsage_User", "NodeCpuStat_TotalUsage_Nice",
                                "NodeCpuStat_TotalUsage_System", "NodeCpuStat_TotalUsage_Idle",
                                "NodeCpuStat_TotalUsage_IoWait", "NodeCpuStat_TotalUsage_Irq",
                                "NodeCpuStat_TotalUsage_SoftIrq", "NodeCpuStat_TotalUsage_Steal",
                                "NodeCpuStat_TotalUsage_UserVar", "NodeCpuStat_TotalUsage_NiceVar",
                                "NodeCpuStat_TotalUsage_SystemVar", "NodeCpuStat_TotalUsage_IdleVar",
                                "NodeCpuStat_TotalUsage_IoWaitVar", "NodeCpuStat_TotalUsage_IrqVar",
                                "NodeCpuStat_TotalUsage_SoftIrqVar", "NodeCpuStat_TotalUsage_StealVar",
                                "NodeCpuStat_LogicalCores",
                                "NodeMemStat_TotalBytes", "NodeMemStat_FreeBytes",
                                "NodeMemStat_AvailableBytes", "NodeMemStat_CachedBytes",
                                "NodeDiskStat_Name", "NodeDiskStat_MajorMinor", "NodeDiskStat_ReadOps",
                                "NodeDiskStat_WriteOps", "NodeDiskStat_BytesRead", "NodeDiskStat_BytesWritten",
                                "NodeDiskStat_ReadTimeSeconds", "NodeDiskStat_WriteTimeSeconds",
                                "NodeDiskStat_IoTimeSeconds", "NodeDiskStat_ReadOpsVar",
                                "NodeDiskStat_WriteOpsVar", "NodeDiskStat_BytesReadVar", "NodeDiskStat_BytesWrittenVar",
                                "NodeDiskStat_ReadTimeSecondsVar", "NodeDiskStat_WriteTimeSecondsVar",
                                "NodeDiskStat_IoTimeSecondsVar",
                                "NodeNetStat_Name", "NodeNetStat_Up", "NodeNetStat_RxBytes", "NodeNetStat_TxBytes",
                                "NodeNetStat_RxPackets", "NodeNetStat_TxPackets", "NodeNetStat_RxBytesVar",
                                "NodeNetStat_TxBytesVar", "NodeNetStat_RxPacketsVar", "NodeNetStat_TxPacketsVar",
                                "NodeUptime", "Label1", "Label2", "Label3"])
    for index, row in df.iterrows():
        ndf.at[index, "TimeStamp"] = row["TimeStamp"]
        ndf.at[index, "Time"] = row["Time"]
        ndf.at[index, "Node"] = row["Node"]
        if isinstance(row["CgroupStats"], float):
            continue
        else:
            cgroup_stats_str = row["CgroupStats"]
            node_stats_str = row["NodeStats"]
            container_net_str = row["ContainerNet"]
            cgroup_stats_str = cgroup_stats_str.replace('null', 'None')
            node_stats_str = node_stats_str.replace('null', 'None')
            container_net_str = container_net_str.replace('null', 'None')
            cgroup_stats_dict = eval(cgroup_stats_str)
            node_stats_dict = eval(node_stats_str)
            container_net_dict = eval(container_net_str)
            for k, v in cgroup_stats_dict.items():
                if isinstance(v, str):
                    ndf.at[index, k] = v
                elif isinstance(v, int):
                    ndf.at[index, k] = v
                elif isinstance(v, float):
                    ndf.at[index, k] = v
                elif isinstance(v, dict):
                    for i, j in v.items():
                        ndf.at[index, k + '_' + i] = j
            for k, v in node_stats_dict.items():
                if isinstance(v, str):
                    ndf.at[index, k] = v
                elif isinstance(v, int):
                    ndf.at[index, k] = v
                elif isinstance(v, float):
                    ndf.at[index, k] = v
                elif isinstance(v, dict):
                    for i, j in v.items():
                        if isinstance(j, str):
                            ndf.at[index, k + '_' + i] = j
                        if isinstance(j, int):
                            ndf.at[index, k + '_' + i] = j
                        if isinstance(j, float):
                            ndf.at[index, k + '_' + i] = j
                        elif isinstance(j, dict):
                            for m, n in j.items():
                                ndf.at[index, k + '_' + i + '_' + m] = n
            for k, v in container_net_dict.items():
                if isinstance(v, str):
                    ndf.at[index, k] = v
                elif isinstance(v, int):
                    ndf.at[index, k] = v
                elif isinstance(v, float):
                    ndf.at[index, k] = v
                elif isinstance(v, dict):
                    for i, j in v.items():
                        ndf.at[index, k + '_' + i] = j
    ndf.drop(['NodeName', 'ContainerInfo', 'Namespace', 'ContainerName', 'ApplicationType', 'NodeDiskStat_Name',
              'NodeDiskStat_MajorMinor', 'NodeNetStat_Name', 'JvmStat_JvmInfo'], axis=1, inplace=True)
    ndf.drop(['Restarts', 'OomKills'], axis=1, inplace=True)
    ndf.drop(['CgroupCpuStat_UsageSeconds', 'CgroupCpuStat_ThrottledTimeSeconds'], axis=1, inplace=True)
    ndf.drop(['CgroupIOStat_ReadOps', 'CgroupIOStat_WriteOps', 'CgroupIOStat_ReadBytes', 'CgroupIOStat_WrittenBytes'],
             axis=1, inplace=True)
    ndf.drop(['CgroupFsStat_CapacityBytes', 'CgroupFsStat_UsedBytes', 'CgroupFsStat_ReservedBytes'], axis=1,
             inplace=True)
    ndf.drop(['CgroupDelays_Cpu', 'CgroupDelays_Disk'], axis=1, inplace=True)
    ndf.drop(['CNetListenopen', 'CNetConnectsSuccessful', 'CNetConnectsFailed', 'CNetRetransmits', 'CNetDrops',
              'CNetConnectionsActive'], axis=1, inplace=True)
    ndf.drop(['CNetSendAsClient', 'CNetSendAsServer', 'CNetRecvAsClient', 'CNetRecvAsServer'], axis=1, inplace=True)
    ndf.drop(['NodeCpuStat_TotalUsage_User', 'NodeCpuStat_TotalUsage_Nice', 'NodeCpuStat_TotalUsage_System',
              'NodeCpuStat_TotalUsage_Idle', 'NodeCpuStat_TotalUsage_IoWait', 'NodeCpuStat_TotalUsage_Irq',
              'NodeCpuStat_TotalUsage_SoftIrq', 'NodeCpuStat_TotalUsage_Steal'], axis=1, inplace=True)
    ndf.drop(['NodeDiskStat_ReadOps', 'NodeDiskStat_WriteOps', 'NodeDiskStat_BytesRead', 'NodeDiskStat_BytesWritten',
              'NodeDiskStat_ReadTimeSeconds', 'NodeDiskStat_WriteTimeSeconds', 'NodeDiskStat_IoTimeSeconds'], axis=1,
             inplace=True)
    ndf.drop(['NodeNetStat_RxBytes', 'NodeNetStat_TxBytes', 'NodeNetStat_RxPackets', 'NodeNetStat_TxPackets'], axis=1,
             inplace=True)
    ndf.drop(['NodeUptime'], axis=1,
             inplace=True)

    return ndf


def process_metric_features(df, interval):
    """
    sort df
    :param df:
    :param interval:
    :return:
    """
    bisort_df = df.sort_values(by=["TimeStamp", "PodName"], ascending=[True, True], ignore_index=True)

    return bisort_df


def round_timestamp(category, df, sample_rate):
    """
    Timestamp may not be set to 5 or 10 rightly, due to the collector.
    The collector should have be amended. The timestamp is
    :param category:
    :param df:
    :param sample_rate:
    :return:
    """
    end_num_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    last_timestamp = 0
    for index, row in df.iterrows():
        node = row["Node"]
        time_str = row["Time"]
        if last_timestamp == 0:
            last_timestamp = row["TimeStamp"]
        else:
            if row["TimeStamp"] - last_timestamp == sample_rate or row["TimeStamp"] - last_timestamp == 0:
                last_timestamp = row["TimeStamp"]
            else:
                print("The row {} for the record of {} on {} doesn't match the {}s".format(index, category, node,
                                                                                           sample_rate))
                print("Its TimeStamp is {}".format(row["TimeStamp"]))
                sys.exit(111)
        if sample_rate == 1:
            return df
        if sample_rate == 5:
            for end_num in end_num_list:
                if time_str.endswith(end_num):
                    idx = time_str.rfind(end_num)
                    if end_num == "1" or end_num == "2" or end_num == "3" or end_num == "4":
                        tmp_str = time_str[:idx] + str.replace(time_str[idx:], end_num, "0")
                        assert len(time_str) == len(tmp_str)
                        tmp_timestamp = datetime.strptime(tmp_str, "%Y-%m-%d %H:%M:%S").timestamp()
                        row["Time"] = tmp_str
                        row["TimeStamp"] = tmp_timestamp
                        print("{}: time {} ends with {} at row {}, reset to {}".format(node, time_str, end_num, index,
                                                                                       tmp_str))

                    elif end_num == "6" or end_num == "7" or end_num == "8" or end_num == "9":
                        tmp_str = time_str[:idx] + str.replace(time_str[idx:], end_num, "5")
                        assert len(time_str) == len(tmp_str)
                        tmp_timestamp = datetime.strptime(tmp_str, "%Y-%m-%d %H:%M:%S").timestamp()
                        row["Time"] = tmp_str
                        row["TimeStamp"] = tmp_timestamp
                        print("{}: time {} ends with {} at row {}, reset to {}".format(node, time_str, end_num, index,
                                                                                       tmp_str))
                    else:
                        continue
        if sample_rate == 10:
            for end_num in end_num_list:
                if time_str.endswith(end_num):
                    idx = time_str.rfind(end_num)
                    if end_num == "1" or end_num == "2" or end_num == "3" or end_num == "4" or end_num == "5" \
                            or end_num == "6" or end_num == "7" or end_num == "8" or end_num == "9":
                        tmp_str = time_str[:idx] + str.replace(time_str[idx:], end_num, "0")
                        assert len(time_str) == len(tmp_str)
                        tmp_timestamp = datetime.strptime(tmp_str, "%Y-%m-%d %H:%M:%S").timestamp()
                        row["Time"] = tmp_str
                        row["TimeStamp"] = tmp_timestamp
                        print("{}: time {} ends with {} at row {}, reset to {}".format(node, time_str, end_num, index,
                                                                                       tmp_str))
                    else:
                        print("****************************")
                        print("timestamp recorded abnormal!")
                        print("****************************")
                        sys.exit(222)
    return df


def generate_one_node_processed_dataframe(category, start_timestamp, end_timestamp, interval, node_path, sample_rate,
                                          data_type="metric"):
    """
    Read one node raw data of a category, unify the data time stamp for different nodes,
    then sort the dataframe by index time. rescale the data depending on the data type. Return the dataframe.
    :param node_path: raw data path
    :param start_timestamp: timestamp
    :param end_timestamp: timestamp    datetime format like "%Y-%m-%d %H:%M:%S"
    :param interval: the duration between two steps
    :param data_type: metric or trace
    :return:
    """
    if data_type == "metric":
        if sys.platform.startswith('win'):
            metric_path = node_path + '\\' + data_type
        elif sys.platform.startswith('linux'):
            metric_path = node_path + '/' + data_type
        # metric_path = node_path + '\\' + data_type
        files, _ = get_subfiles_subfolders(metric_path)  #
        count = len(files)
        node_df_list = []

        for i in range(count):
            if files[i].endswith(".log"):
                df = read_metric_from_log(files[i])
                node_df_list.append(df)
            else:
                continue
        df = pd.concat(node_df_list, ignore_index=True)
        flatten_df = flatten_metric_data(df, interval)
        if flatten_df.columns[0] != "TimeStamp":
            flatten_df.drop(flatten_df.columns[0], axis=1, inplace=True)
        processed_df = process_metric_features(flatten_df, interval)
        time_check_df = round_timestamp(category, processed_df, sample_rate)
        new_df = time_check_df[
            (time_check_df['TimeStamp'] >= start_timestamp) & (time_check_df['TimeStamp'] <= end_timestamp)]
        if sys.platform.startswith('win'):
            new_df.to_csv(metric_path + "\\new_df.csv", header=True, index=False)
        elif sys.platform.startswith('linux'):
            new_df.to_csv(metric_path + "/new_df.csv", header=True, index=False)
        # new_df.to_csv(metric_path + "\\new_df.csv", header=True, index=False)
        # index_df = bisort_df.set_index("TimeStamp", drop=True)
        # sorted_df = index_df.sort_index()
        return new_df

    elif data_type == "trace":
        if sys.platform.startswith('win'):
            trace_path = node_path + '\\' + data_type
        elif sys.platform.startswith('linux'):
            trace_path = node_path + '/' + data_type
        # trace_path = node_path + '\\' + data_type
        all_files, _ = get_subfiles_subfolders(trace_path)
        trace_files = []
        for file in all_files:
            if file.find("network_state_trace") != -1 and file.endswith("log"):
                trace_files.append(file)
        count = len(trace_files)
        trace_df_list = []
        for i in range(count):
            if trace_files[i].endswith(".log"):
                df = read_trace_from_log(trace_files[i])
                if df.empty:
                    print("{} is empty file.".format(trace_files[i]))
                else:
                    trace_df_list.append(df)
            else:
                continue
        if len(trace_df_list) == 0:
            new_df = pd.DataFrame()
        else:
            df = pd.concat(trace_df_list)
            new_df = df[(df['TimeStamp'] >= start_timestamp) & (df['TimeStamp'] <= end_timestamp)]
            if sys.platform.startswith('win'):
                new_df.to_csv(trace_path + "\\new_df.csv", header=True, index=False)
            elif sys.platform.startswith('linux'):
                new_df.to_csv(trace_path + "/new_df.csv", header=True, index=False)
            # new_df.to_csv(trace_path + "\\new_df.csv", header=True, index=False)
        return new_df


def get_one_node_log_timestamp(node_path):
    """
    node path may have log files crossing one day
    find the start stamp on previous day and end stamp on next day
    timestamp is 19bit, need to divide by 1000000000, and transformer to datatime type, has no time zone info.
    "%Y-%m-%d %H:%M:%S"    not "%Y-%m-%d %H:%M:%S%z"
    :param node_path:
    :return:
    """
    if sys.platform.startswith('win'):
        node_metric_path = node_path + '\\' + 'metric'
    elif sys.platform.startswith('linux'):
        node_metric_path = node_path + '/' + 'metric'
    # log_files, _ = get_subfiles_subfolders(node_path + '\\' + 'metric')
    log_files, _ = get_subfiles_subfolders(node_metric_path)
    first_start_stamp = 0
    last_end_stamp = 0
    for log in log_files:
        if log.endswith("log"):
            df = read_metric_from_log(log)
            start_time = df.iloc[0]["Time"]
            end_time = df.iloc[-1]["Time"]
            start_stamp = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp()
            end_stamp = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp()
            if first_start_stamp == 0:
                first_start_stamp = start_stamp
            elif first_start_stamp > start_stamp:
                first_start_stamp = start_stamp
            elif 0 < first_start_stamp < start_stamp:
                pass
            if last_end_stamp == 0:
                last_end_stamp = end_stamp
            elif last_end_stamp < end_stamp:
                last_end_stamp = end_stamp
            elif last_end_stamp > end_stamp:
                pass
    return first_start_stamp, last_end_stamp


def generate_label_by_fault_injection_type(category, df):
    """
    :param category:
    :param df:
    :return:
    """
    for index, row in df.iterrows():
        time_str = row["Time"]
        clock_str = time_str.split(" ")[1]
        minute_str = clock_str.split(":")[1]
        minute_int = int(minute_str)
        pod = row["PodName"]
        if category.find("normal"):
            df.at[index, "Label1"] = 0
            df.at[index, "Label2"] = 0
            df.at[index, "Label3"] = 0
        elif category.find("network-loss"):
            df.at[index, "Label1"] = 1
            if (10 <= minute_int <= 13) or (30 <= minute_int <= 33) or (50 <= minute_int <= 53):
                if pod.find("ts-contacts-service") != -1 or pod.find("ts-seat-service") != -1 \
                        or pod.find("ts-route-service") != -1:
                    df.at[index, "Label2"] = 1
                    df.at[index, "Label3"] = 1
                else:
                    df.at[index, "Label2"] = 0
                    df.at[index, "Label3"] = 0
            else:
                df.at[index, "Label2"] = 0
                df.at[index, "Label3"] = 0
        elif category.find("network-delay"):
            df.at[index, "Label1"] = 1
            if (0 <= minute_int <= 3) or (20 <= minute_int <= 23) or (40 <= minute_int <= 43):
                if pod.find("ts-contacts-service") != -1 or pod.find("ts-seat-service") != -1 \
                        or pod.find("ts-route-service") != -1:
                    df.at[index, "Label2"] = 1
                    df.at[index, "Label3"] = 2
                else:
                    df.at[index, "Label2"] = 0
                    df.at[index, "Label3"] = 0
            else:
                df.at[index, "Label2"] = 0
                df.at[index, "Label3"] = 0
        elif category.find("network-corrupt"):
            df.at[index, "Label1"] = 1
            if (10 <= minute_int <= 13) or (30 <= minute_int <= 33) or (50 <= minute_int <= 53):
                if pod.find("ts-contacts-service") != -1 or pod.find("ts-seat-service") != -1 \
                        or pod.find("ts-route-service") != -1:
                    df.at[index, "Label2"] = 1
                    df.at[index, "Label3"] = 3
                else:
                    df.at[index, "Label2"] = 0
                    df.at[index, "Label3"] = 0
            else:
                df.at[index, "Label2"] = 0
                df.at[index, "Label3"] = 0
        elif category.find("network-bandwidth"):
            df.at[index, "Label1"] = 1
            if (0 <= minute_int <= 3) or (10 <= minute_int <= 13) or (20 <= minute_int <= 23) \
                    or (30 <= minute_int <= 33) or (40 <= minute_int <= 43) or (50 <= minute_int <= 53):
                if pod.find("ts-contacts-service") != -1 or pod.find("ts-seat-service") != -1 \
                        or pod.find("ts-route-service") != -1:
                    df.at[index, "Label2"] = 1
                    df.at[index, "Label3"] = 4
                else:
                    df.at[index, "Label2"] = 0
                    df.at[index, "Label3"] = 0
            else:
                df.at[index, "Label2"] = 0
                df.at[index, "Label3"] = 0
        elif category.find("cpu-stress"):
            df.at[index, "Label1"] = 1
            if (0 <= minute_int <= 3) or (10 <= minute_int <= 13) or (20 <= minute_int <= 23) \
                    or (30 <= minute_int <= 33) or (40 <= minute_int <= 43) or (50 <= minute_int <= 53):
                if pod.find("ts-contacts-service") != -1:
                    df.at[index, "Label2"] = 1
                    df.at[index, "Label3"] = 5
                else:
                    df.at[index, "Label2"] = 0
                    df.at[index, "Label3"] = 0
            else:
                df.at[index, "Label2"] = 0
                df.at[index, "Label3"] = 0
        elif category.find("mem-stress"):
            df.at[index, "Label1"] = 1
            if (0 <= minute_int <= 3) or (10 <= minute_int <= 13) or (20 <= minute_int <= 23) \
                    or (30 <= minute_int <= 33) or (40 <= minute_int <= 43) or (50 <= minute_int <= 53):
                if pod.find("ts-contacts-service") != -1 or pod.find("ts-seat-service") != -1:
                    df.at[index, "Label2"] = 1
                    df.at[index, "Label3"] = 6
                else:
                    df.at[index, "Label2"] = 0
                    df.at[index, "Label3"] = 0
            else:
                df.at[index, "Label2"] = 0
                df.at[index, "Label3"] = 0
        elif category.find("io-latency"):
            df.at[index, "Label1"] = 1
            if (0 <= minute_int <= 3) or (10 <= minute_int <= 13) or (20 <= minute_int <= 23) \
                    or (30 <= minute_int <= 33) or (40 <= minute_int <= 43) or (50 <= minute_int <= 53):
                if pod.find("tsdb-mysql-1") != -1 or pod.find("tsdb-mysql-2") != -1:
                    df.at[index, "Label2"] = 1
                    df.at[index, "Label3"] = 7
                else:
                    df.at[index, "Label2"] = 0
                    df.at[index, "Label3"] = 0
            else:
                df.at[index, "Label2"] = 0
                df.at[index, "Label3"] = 0
        else:
            print("Unkonwn category! Need to Process")
            sys.exit(1000)

    return df


def check_number_of_instances_for_each_step(df, sample_rate):
    group_df = df.groupby("TimeStamp").count()
    total_num_instances = group_df.iloc[0]["Time"]
    num_instances_of_vm_list = []
    last_count = 0

    for index, row in group_df.iterrows():
        if last_count == 0:
            last_count = row["Time"]
        else:
            if last_count == row["Time"]:
                last_count = row["Time"]
                # print("row {} and row {} equal".format(index - sample_rate, index))
                continue
            else:
                print("Find unequal number instance on two steps")
                return False, -1, []

    all_step_df = df.groupby("TimeStamp")
    for ts, one_step_df in all_step_df:
        first_df = one_step_df
        break
    first_step_df = first_df.groupby("Node").count()
    for index, row in first_step_df.iterrows():
        num_instance_count = row["Time"]
        num_instances_of_vm_list.append(num_instance_count)

    return True, total_num_instances, num_instances_of_vm_list


def read_ip_list_from_category(category_path):
    ip2pod_dict = {}
    clusterip2svc_dict = {}
    if sys.platform.startswith('win'):
        ip_list_file_path = category_path + "\\" + "ip_list.txt"
        clusterip_list_file_path = category_path + "\\" + "clusterip_list.txt"
    elif sys.platform.startswith('linux'):
        ip_list_file_path = category_path + "/" + "ip_list.txt"
        clusterip_list_file_path = category_path + "/" + "clusterip_list.txt"
    # ip_list_file_path = category_path + "\\" + "ip_list.txt"
    # clusterip_list_file_path = category_path + "\\" + "clusterip_list.txt"
    if os.path.exists(ip_list_file_path):
        with open(ip_list_file_path, "r") as f:
            txt_lines = f.readlines()
            i = 0
            for line in txt_lines:
                if i == 0:
                    i += 1
                    continue
                fileds = line.split(":")
                pod_name = fileds[0].split(".")[1]
                ip = fileds[1].replace('\n', '')
                ip2pod_dict[ip] = pod_name
                i += 1

    if os.path.exists(clusterip_list_file_path):
        with open(clusterip_list_file_path, "r") as f:
            txt_lines = f.readlines()
            i = 0
            for line in txt_lines:
                if i == 0:
                    i += 1
                    continue
                fileds = line.split(":")
                svc_name = fileds[0].split(".")[1]
                clusterip = fileds[1].replace('\n', '')
                clusterip2svc_dict[clusterip] = svc_name
                i += 1
    if len(ip2pod_dict) == 0:
        print("ip_list file does not exist")
        sys.exit(666)
    elif len(clusterip2svc_dict) == 0:
        print("clusterip_list file does not exist")
        sys.exit(666)
    return ip2pod_dict, clusterip2svc_dict


def generate_instance_map(df, num_instances):
    name2index_dict = {}
    index2name_dict = {}
    new_df = df.iloc[:num_instances]
    for index, row in new_df.iterrows():
        name2index_dict[row["Node"] + "-" + row["PodName"]] = index
        index2name_dict[index] = row["Node"] + "-" + row["PodName"]
    return name2index_dict, index2name_dict


def generate_category_dataframe(category, category_path, node_folders, latest_start_stamp, earliest_end_stamp,
                                sample_rate):
    """

    :param category:
    :param category_path:
    :param node_folders:
    :param latest_start_stamp:
    :param earliest_end_stamp:
    :param sample_rate:
    :return:
    """
    node_metric_df_list = []
    for node_path in node_folders:
        if node_path.find("output") == -1:
            df = generate_one_node_processed_dataframe(category, latest_start_stamp, earliest_end_stamp, 12,
                                                       node_path, sample_rate, data_type="metric")
            node_metric_df_list.append(df)
    all_node_df = pd.concat(node_metric_df_list)
    all_node_sort_df = all_node_df.sort_values(by=["TimeStamp", "Node", "PodName"], ignore_index=True)
    # all_node_sort_df.to_csv(category_path + category + "_all_nodes_metric_sort_df" + ".csv", header=True, index=False)
    labeled_df = generate_label_by_fault_injection_type(category, all_node_sort_df)
    is_all_match, total_num_instances, num_instances_of_vm_list = check_number_of_instances_for_each_step(labeled_df, sample_rate)
    print("The number of monitoring instances for {} is {}".format(category, total_num_instances))
    if is_all_match:
        pod2index_dict, index2pod_dict = generate_instance_map(labeled_df, total_num_instances)
    else:
        print("some time steps have different number reocrded instances!!!")
        sys.exit(444)
    labeled_df.to_csv(category_path + category + "_all_nodes_metric_labeled_df" + ".csv", header=True, index=False)
    instances_index_vm_df = pd.DataFrame(data=num_instances_of_vm_list, columns=["NumInstances"])
    instances_index_vm_df.to_csv(category_path + category + "_instances_index_vm_df" + ".csv", header=True, index=False)

    return labeled_df, pod2index_dict, index2pod_dict


def generate_span_connection_dataframe(span_df, start_timestamp, end_timestamp, sample_rate, ip2pod_dict,
                                       clusterip2svc_dict, pod2index_dict):
    new_df = pd.DataFrame(columns=["TimeStamp", "Time", "From", "To", "ConnectLatNs", "Count"])
    group_df = span_df.groupby(["SaddrV4", "DaddrV4"])
    row_index = 0
    for (src, dst), df in group_df:
        From = -1
        To = -1
        if src in ip2pod_dict.keys():
            FromPod = ip2pod_dict[src]  # only pod name
            for vmpod in pod2index_dict.keys():  # vmpod is like "node1-[podname]"
                if vmpod.find(FromPod) != -1:
                    From = pod2index_dict[vmpod]
                    break
                else:
                    continue
        elif src in clusterip2svc_dict.keys():
            svc = ip2pod_dict[src]  # only short name of pod name
            for vmpod in pod2index_dict.keys():  # vmpod is like "node1-[podname]"
                # only has one instance for the svc, it's simple to handle
                # but multiple instances need another way to process
                if vmpod.find(svc) != -1:
                    From = pod2index_dict[vmpod]
                    break
                else:
                    continue
        else:
            print("Couldn't match the src ip address {} to index".format(src))

        if dst in ip2pod_dict.keys():
            ToPod = ip2pod_dict[dst]  # only pod name
            for vmpod in pod2index_dict.keys():  # vmpod is like "node1-[podname]"
                if vmpod.find(ToPod) != -1:
                    To = pod2index_dict[vmpod]
                    break
                else:
                    continue
        elif dst in clusterip2svc_dict.keys():
            svc = ip2pod_dict[dst]  # only short name of pod name
            for vmpod in pod2index_dict.keys():  # vmpod is like "node1-[podname]"
                # only has one instance for the svc, it's simple to handle
                # but multiple instances need another way to process
                if vmpod.find(svc) != -1:
                    To = pod2index_dict[vmpod]
                    break
                else:
                    continue
        else:
            print("Couldn't match the dst ip address {} to index".format(dst))

        if From == -1 or To == -1:
            print("Couldn't match the ip address from {} to {} to index".format(src, dst))
            sys.exit(555)
        new_df.at[row_index, "From"] = From
        new_df.at[row_index, "To"] = To
        new_df.at[row_index, "ConnectLatNs"] = df["ConnectLatMs"].mean()
        new_df.at[row_index, "Count"] = df["ConnectLatMs"].count()
        new_df.at[row_index, "TimeStamp"] = span_df.iloc[row_index]["TimeStamp"]
        new_df.at[row_index, "Time"] = span_df.iloc[row_index]["Time"]
        row_index += 1
    return new_df


def generate_category_connection(category, category_path, node_folders, latest_start_stamp, earliest_end_stamp,
                                 sample_rate, ip2pod_dict, clusterip2svc_dict, pod2index_dict):
    node_trace_df_list = []
    for node_path in node_folders:
        if node_path.find("output") == -1:
            df = generate_one_node_processed_dataframe(category, latest_start_stamp, earliest_end_stamp, 12,
                                                       node_path, sample_rate, data_type="trace")
            if df.empty:
                continue
            else:
                node_trace_df_list.append(df)
    all_trace_df = pd.concat(node_trace_df_list)
    all_node_sort_df = all_trace_df.sort_values(by=["TimeStamp"], ignore_index=True)
    all_node_sort_df.to_csv(category_path + category + "_all_node_trace_sort_df" + ".csv", header=True, index=False)

    # timestamp2df_dict = {}
    timestamp2df_list = []
    tmp_timestamp = latest_start_stamp - sample_rate
    while tmp_timestamp + sample_rate <= earliest_end_stamp:
        span_df = all_node_sort_df[
            (all_node_sort_df['TimeStamp'] >= tmp_timestamp) & (
                    all_node_sort_df['TimeStamp'] <= tmp_timestamp + sample_rate)]
        step_df = generate_span_connection_dataframe(span_df, tmp_timestamp, tmp_timestamp + sample_rate, sample_rate,
                                                     ip2pod_dict, clusterip2svc_dict, pod2index_dict)
        # timestamp2df_dict[tmp_timestamp + sample_rate] = step_df
        timestamp2df_list.append(step_df)
        tmp_timestamp = tmp_timestamp + sample_rate

    all_connection_df = pd.concat(timestamp2df_list)
    all_connection_df.to_csv(category_path + category + "_all_connection_df" + ".csv", header=True)
    return all_connection_df


def process_metric_trace_raw_data():
    if sys.platform.startswith('win'):
        base_path = "./"
    elif sys.platform.startswith('linux'):
        base_path = '.'
    # base_path = "./"
    category_list = scan_category(base_path)
    metric_df_1_list = []
    metric_df_5_list = []
    metric_df_10_list = []
    trace_df_1_list = []
    trace_df_5_list = []
    trace_df_10_list = []
    processed_list = ["cpu-stress30-15-1h-5s", "cpu-stress30-20-1h-5s", "io-latency200-15-1h-5s",
                      "io-latency200-20-1h-5s", "mem-stress30-15-1h-5s", "mem-stress30-20-1h-5s",
                      "network-corrupt15-15-2h-5s", "network-corrupt20-15-2h-5s", "network-delay200-15-2h-5s",
                      "network-delay200-20-2h-5s", "network-loss15-05-2h-5s", "network-loss15-15-2h-5s",
                      "network-loss15-25-2h-5s", "normal-05-2h-5s", "normal-15-2h-5s", "normal-25-2h-5s"]

    processed_list = ['cpu-stress30-15-1h-5s', 'cpu-stress30-20-1h-5s', 'io-latency200-15-1h-5s',
                      'io-latency200-20-1h-5s', 'mem-stress30-15-1h-5s', 'mem-stress30-20-1h-5s',
                      'network-corrupt15-15-2h-5s', 'network-corrupt20-15-2h-5s', 'network-delay200-15-2h-5s',
                      ]
    for category in tqdm.tqdm(category_list):
        if category in processed_list:
            continue
        else:
            print("-"*100)
            print("Beginning to process {}".format(category))
            if sys.platform.startswith('win'):
                category_path = './' + category + '/'
            elif sys.platform.startswith('linux'):
                category_path = category + '/'
            _, node_folders = get_subfiles_subfolders(category_path)
            start_stamp_list = []
            end_stamp_list = []
            for node_path in node_folders:
                if node_path.find("output") == -1:
                    start_stamp, end_stamp = get_one_node_log_timestamp(node_path)
                    start_stamp_list.append(start_stamp)
                    end_stamp_list.append(end_stamp)
                else:
                    continue
            sorted_start_stamp_list = sorted(start_stamp_list)
            sorted_end_stamp_list = sorted(end_stamp_list)
            if category_path.find("-1s") != -1:
                trim_delta = 6 * 1
                sample_rate = 1
            elif category_path.find("-5s") != -1:
                trim_delta = 6 * 5
                sample_rate = 5
            elif category_path.find("-10s") != -1:
                trim_delta = 6 * 10
                sample_rate = 10
            latest_start_stamp = int(sorted_start_stamp_list[-1]) + trim_delta
            earliest_end_stamp = int(sorted_end_stamp_list[0]) - trim_delta

            df, pod2index_dict, index2pod_dict = generate_category_dataframe(category, category_path, node_folders,
                                                                             latest_start_stamp, earliest_end_stamp,
                                                                             sample_rate)
            if category.find("-1s") != -1:
                metric_df_1_list.append(df)
            elif category.find("-5s") != -1:
                metric_df_5_list.append(df)
            elif category.find("-10s") != -1:
                metric_df_10_list.append(df)

            ip2pod_dict, clusterip2svc_dict = read_ip_list_from_category(category_path)
            all_connection_df = generate_category_connection(category, category_path, node_folders,
                                                             latest_start_stamp, earliest_end_stamp,
                                                             sample_rate, ip2pod_dict,
                                                             clusterip2svc_dict, pod2index_dict)
            print("The data of {} has been processed successfully!".format(category))


if __name__ == "__main__":
    process_metric_trace_raw_data()
    