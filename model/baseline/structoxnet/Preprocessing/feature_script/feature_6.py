#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np


def cal_dssp_feature( seq_id):

    file_path = './outputs/dssp/{}.dssp'.format( seq_id)
    with open(file_path, 'r') as fd:
        fdlines = fd.readlines()

    length = int(fdlines[-1].split()[0])
    feature_list = [[0 for _ in range(6)] for _ in range(length)] # 1D feature dimension is set 6, may change


    for linei in range(2, len(fdlines)-1):
            line = fdlines[linei].split()
            line_prev = fdlines[linei-1].split()
            line_post = fdlines[linei+1].split()
            
            res = int(line[0]) - 1
            cai = [float(line[9]), float(line[10]), float(line[11])]
            cai_prev = [float(line_prev[9]), float(line_prev[10]), float(line_prev[11])]
            cai_post = [float(line_post[9]), float(line_post[10]), float(line_post[11])]
            #提取三个Ca的坐标信息
            
            forw = np.subtract(cai_post, cai) #坐标对应元素相减
            revw = np.subtract(cai_prev, cai)

            forwn = math.sqrt(forw[0] ** 2 + forw[1] ** 2 +  forw[2] ** 2) #计算向量的模长
            revwn = math.sqrt(revw[0] ** 2 + revw[1] ** 2 +  revw[2] ** 2)

            u_forw = forw/forwn
            u_revw = revw/revwn
            #print(u_forw, u_revw)

            feature_list[res][:3] = u_forw 
            feature_list[res][3:] = u_revw
            #print(feature_list[res])
    return feature_list





