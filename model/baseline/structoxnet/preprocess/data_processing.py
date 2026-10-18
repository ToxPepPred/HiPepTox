# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 9:00
# @Author  : Jiao Shihu
# @Email   : shihujiao@163.com
# @IDE     : PyCharm
# @FileName: data_processing.py



from functools import partial
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader



def prepare_data_list(datafile):
    data_list = []
    for key, value in datafile.items():
        label = float(int(key.split('_')[-1]))
        data = Data(x=value[0], nodes=value[1], edge_index=value[2], edge_s=value[3],edge_v=value[4], plm=value[5], label=label)
        data_list.append(data)
    
    return data_list


def create_data_loaders(test_data, config):
    test_list = prepare_data_list(test_data)
    dataloader = partial(DataLoader, num_workers=4, batch_size=config.batch_size, shuffle=True, drop_last=False)
    test_loader = dataloader(test_list, shuffle=False, drop_last=False)

    return test_loader


