#!/usr/bin/env python
# coding: utf-8

# ##  每个氨基酸的内部几何特征，由CA、N、C原子坐标计算，只需要pdb文件计算，3维。描述蛋白质主链中每个残基局部几何结构的特征向量。具体来说，它捕捉了N、CA 和 C 三个原子形成的局部三角形的空间形状和方向。



import math
import numpy as np
from scipy.spatial import ConvexHull




def get_tetrahedral_geom(pos): # pos包含pdb所有原子有效信息
    last_res_no = int(pos[-1][22:(22+4)].strip()) # int(seq_len)

    thg = [[[0,0,0],[0,0,0],[0,0,0]] for _ in range(last_res_no)] # seq_len*3*3的全0
    thg_val = [[0, 0, 0] for _ in range(last_res_no)] # seq_len*3的全0

    for i in range(len(pos)):
        res_no = int(pos[i][22:(22+4)].strip()) #residue的编号
        atom_type = pos[i][13:(13+2)].strip() # 原子的类型

        xyz = [float(pos[i][30:(30+8)]), float(pos[i][38:(38+8)]), float(pos[i][46:(46+8)])]
        if(atom_type == 'CA'):
                thg[res_no-1][0] = xyz
        elif(atom_type == 'C'):
                thg[res_no-1][1] = xyz
        elif(atom_type == 'N'):
                thg[res_no-1][2] = xyz

    for i in range(len(thg_val)):
        N = np.array(thg[i][2])
        Ca = np.array(thg[i][0])
        C = np.array(thg[i][1])
        n = N - Ca # Ca到N的向量
        c = C - Ca # Ca到C的向量
        cross = np.cross(n,c) #计算两个向量的交叉，得到垂直于 N、Ca 和 C 原子所形成的平面的向量
        t1 = cross/((cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) * math.sqrt(3)) # 归一化
        #summ = [n[0] + c[0], n[1] + c[1], n[2] + c[2]]
        summ = n + c
        t2 = math.sqrt(2/3) * summ / (summ[0] ** 2 + summ[1] ** 2 + summ[2] ** 2)
        thg_val[i] = t1 - t2 #[t1[0] - t2[0],  t1[1] - t2[1], t1[2] - t2[2]]
        if(np.isnan(thg_val[i]).any()):
            thg_val[i] = [0,0,0]
    return thg_val


# In[15]:


def get_residue_area_volume(pos):
    last_res_no = int(pos[-1][22:(22+4)].strip())
    res_coords = [[] for _ in range(last_res_no)]
    area = [0 for _ in range(last_res_no)]
    vol = [0 for _ in range(last_res_no)]
    #print(res_coords)
    for i in range(len(pos)):
        res_no = int(pos[i][22:(22+4)].strip())
        res_coords[res_no-1].append([float(pos[i][30:(30+8)]), float(pos[i][38:(38+8)]), float(pos[i][46:(46+8)])])

    for index, res in enumerate(res_coords):
        #print(res)
        if(len(res) < 4):
                continue
        hull = ConvexHull(res)
        vol[index] = hull.volume
        area[index] = hull.area

    return area, vol


# In[16]:


def comput_gem_fea(seq_id, seq):
    length = len(seq)
    file_path = './inputs/pdb/{}.pdb'.format(seq_id)

    pos = []
    with open(file_path, 'r') as fm:
        for line in fm:
            if line.startswith('ATOM'):
                pos.append(line.strip())

    dhg_val = get_tetrahedral_geom(pos)
    areaA, volA = get_residue_area_volume(pos)
    if len(dhg_val) != length or len(areaA) != length or len(volA) != length:
        raise ValueError("Lengths of dhg_val, areaA, and volA do not match the sequence length.")
    
    combined_features = []
    for i in range(length):
        dhg_val_flat = dhg_val[i].tolist()  
#         combined_element = dhg_val_flat + [areaA[i]] + [volA[i]]
        combined_element = dhg_val_flat + [areaA[i]]
        combined_features.append(combined_element)
    
    return combined_features




