#!/usr/bin/env python
# coding: utf-8

# In[31]:


import math
import numpy as np



def dist(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)


# In[23]:


ss_mapping = {'H': 0, 'G': 1, 'I': 2, 'E': 3, 'B': 4, 'T': 5, 'S': 6}
sa_thresholds = [30, 60, 90, 120, 150, 180, 210]


def ss_one_hot8(ss):
    one_hot = np.zeros(8)
    one_hot[ss_mapping.get(ss, 7)] = 1
    return tuple(one_hot)


# In[25]:


def sa_one_hot8(sa):
    sa = float(sa)
    one_hot = np.zeros(8)
    index = np.searchsorted(sa_thresholds, sa)
    one_hot[index] = 1
    return tuple(one_hot)


# In[26]:


def centroid(pos):
    cx = 0
    cy = 0
    cz = 0
    for i in range(len(pos)):
        cx += pos[i][0]
        cy += pos[i][1]
        cz += pos[i][2]
    cx /= len(pos)
    cy /= len(pos)
    cz /= len(pos)
    return cx, cy, cz


# ## 距离质心的逆值 (1 维), 二级结构, 溶剂可及性的, 每个特征值除以 360 进行归一化, 1+8+8+5= 22


def cal_dssp_feature( seq_id):
    file_path = './outputs/dssp/{}.dssp'.format(seq_id)
    with open(file_path, 'r') as fd:
        fdlines = fd.readlines()

    length = int(fdlines[-1].split()[0])

    dssp_list = [[0 for _ in range(22)] for _ in range(length)]

    pos = []#这里的pos包含所有的Ca
    for line in fdlines[1:]:
        line = line.strip().split()  
        x = float(line[9])
        y = float(line[10])
        z = float(line[11])
        pos.append([x, y, z])

    cx, cy, cz = centroid(pos)


    for line in fdlines[1:]:
        line_data = line.strip().split()
        index = int(line_data[0]) - 1
        x = float(line_data[9])
        y = float(line_data[10])
        z = float(line_data[11])
        d = dist(x, y, z, cx, cy, cz)  
        d = 1/d
        #print(d)
        dssp_list[index][0] = d #for d
        dssp_list[index][1:(1+8)] = ss_one_hot8(line_data[2])
        dssp_list[index][9:(9+8)] = sa_one_hot8(line_data[3])
        dssp_list[index][17] = float(line_data[4]) 
        dssp_list[index][18] = float(line_data[5])/360.0 
        dssp_list[index][19] = float(line_data[6])/360.0
        dssp_list[index][20] = float(line_data[7])/360.0  
        dssp_list[index][21] = float(line_data[8])/360.0    

    return dssp_list




