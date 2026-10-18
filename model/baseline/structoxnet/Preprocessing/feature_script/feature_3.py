#!/usr/bin/env python
# coding: utf-8

amino_acids = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 'I':9, 'L':10, 'K':11, 'M':12, 
               'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19}



def ss_one_hot3(ss):
    if(ss == 'H' or ss == 'G' or ss == 'I'): #helix
            return 1, 0, 0
    elif(ss == 'E' or ss == 'B'):
            return 0, 1, 0
    else:
            return 0, 0, 1

def sa_one_hot2(sa):
    if(float(sa) > 50):
            return 1, 0 #solvant acc
    else:
            return 0, 1


# 二级结构（helix, sheet, coil）,溶剂可及性（暴露或埋藏） 3+2=5，是dssp特征, 1 (相对位置) ，总共6维


def cal_dssp_feature(seq_id):
    file_path = './outputs/dssp/{}.dssp'.format(seq_id)
    with open(file_path, 'r') as fd:
        fdlines = fd.readlines()

    length = int(fdlines[-1].split()[0]) # seq_len
    
    dssp_list = [[0 for _ in range(5)] for _ in range(length)]
     
    for line in fdlines[1:]:
        line = line.strip().split()
        if(line[1] not in amino_acids):
                print('not in aminoacids')
                print(line[1])

        ss_features = list(ss_one_hot3(line[2]))
        sa_features = list(sa_one_hot2(line[3]))
        dssp_list[int(line[0]) - 1] = ss_features + sa_features

    feature_list = [dssp_features + [1/(i+1)] for i, dssp_features in enumerate(dssp_list)]
    
    return feature_list



