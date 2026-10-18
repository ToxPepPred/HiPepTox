# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 9:00
# @Author  : Jiao Shihu
# @Email   : shihujiao@163.com
# @IDE     : PyCharm
# @FileName: config.py


import argparse

def get_train_config():
    parse = argparse.ArgumentParser(description=' train model')
    parse.add_argument('-cuda', type=bool, default=True, help='whether to use CUDA for GPU acceleration')
    parse.add_argument('-device', type=str, default="cuda", help='device to use: "cuda" for GPU, "cpu" for CPU')



    parse.add_argument('-batch-size', type=int, default=512, help='number of samples per batch')
    parse.add_argument('-max-len', type=int, default=50, help='maximum length of input sequences')
    parse.add_argument('-num-layer', type=int, default=3, help='number of encoder blocks')
    parse.add_argument('-dim-embedding', type=int, default=256, help='dimension of the embedding vectors')
    parse.add_argument('-in_edge_nf', type=int, default=16, help='input dimension of edge features')
    parse.add_argument('-dropout', type=float, default=0.5, help='dropout rate for regularization')

    parse.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    config = parse.parse_args()

    return config

