# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 9:00
# @Author  : Jiao Shihu
# @Email   : shihujiao@163.com


from feature_script import feature_1
from feature_script import feature_2
from feature_script import feature_3
from feature_script import feature_4
from feature_script import feature_5
from feature_script import feature_6
import pickle
import argparse

# In[2]:


def read_fasta_file(filepath):
    with open(filepath, "r") as file:  
        lines = [line.strip() for line in file]
        sequences_dict = {lines[i][1:]: lines[i+1] for i in range(0, len(lines), 2)}
    return sequences_dict




test_sequences = read_fasta_file("./inputs/test_sequences.txt")


# In[4]:


def compute_features(sequences):
    all_feature = {}
    for seq_id in sequences:
        features_1 = feature_1.comput_gem_fea(seq_id, sequences[seq_id])
        features_2 = feature_2.comput_gem_fea(seq_id, sequences[seq_id])
        features_3 = feature_3.cal_dssp_feature(seq_id)
        features_4 = feature_4.cal_dssp_feature(seq_id)
        features_5 = feature_5.cal_dssp_feature(seq_id)
        features_6 = feature_6.cal_dssp_feature(seq_id)


        lengths = [len(features_1), len(features_2), len(features_3), len(features_4), len(features_5), len(features_6)]
        if len(set(lengths)) != 1:
            raise ValueError(f"Feature lengths do not match for test_id: {seq_id}. Lengths: {lengths}")

        combined_feature = []
        for i in range(len(features_1)):
            combined_element = features_1[i] + features_2[i] + features_3[i] + features_4[i] + features_5[i] + features_6[i]
            combined_feature.append(combined_element)
        all_feature[seq_id] = combined_feature
        
    return all_feature




if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.",description="Compute structure-based node features")
    parser.add_argument("-i", required=True, help="Path to the FASTA file containing sequences.")
    args = parser.parse_args()

    test_sequences = read_fasta_file(args.i)
    test_features = compute_features(test_sequences)

    with open('./outputs/test_features.pkl', 'wb') as pklfile:
        pickle.dump(test_features, pklfile)




