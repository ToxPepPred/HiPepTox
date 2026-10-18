#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math





def cal_dssp_feature( seq_id):

    file_path = './outputs/dssp/{}.dssp'.format(seq_id)

    with open(file_path, 'r') as fd:
        fdlines = fd.readlines()

    length = int(fdlines[-1].split()[0])
    feature_list = [[0 for _ in range(6)] for _ in range(length)] 


    for line in fdlines[1:]:
        line = line.strip().split()
        res = int(line[0]) - 1
        alpha = float(line[6])
        phi = float(line[7])  
        psi = float(line[8])
        alphasin = math.sin(math.radians(alpha))
        alphacos = math.cos(math.radians(alpha))
        phisin = math.sin(math.radians(phi))
        phicos = math.cos(math.radians(phi))
        psisin = math.sin(math.radians(psi))
        psicos = math.cos(math.radians(psi))
        feature_list[res] = [alphasin, alphacos, phisin, phicos, psisin, psicos]
    
    return feature_list






