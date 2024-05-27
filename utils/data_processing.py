import pandas as pd


def get_vm_instances(filepath, traversal_limit=10):
    """
    Counting the number of unique instances on a vm
    :param filepath: processed data csv file path before split into train val test dataset
    :param traversal_limit: traversal limit to limit the search complexity
    :return: the number of instances on various vms
    """
    print("Counting the number of instances deployed on vm...")
    df = pd.read_csv(filepath)
    vm_instances_id = {}
    vm_instances_dic = {}
    vm_instances_count_list = []
    traversal_count = 0
    for index, row in df.iterrows():
        traversal_count += 1
        if traversal_count == traversal_limit:
            break
        row_id = row["Node"] + "-" + row["PodName"]
        vm_name = row["Node"]
        if row_id not in vm_instances_id.keys():
            vm_instances_id[row_id] = 0
            if vm_name not in vm_instances_dic.keys():
                vm_instances_dic[vm_name] = 1
            else:
                vm_instances_dic[vm_name] += 1
        else:
            break

    for i in range(len(vm_instances_dic)):
        vm_instances_count_list.append(vm_instances_dic["node" + str(i)])

    if len(vm_instances_count_list) > 0:
        print("Counting completed!")
    return vm_instances_count_list, traversal_count


