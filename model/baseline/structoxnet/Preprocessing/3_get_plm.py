# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 9:00
# @Author  : Jiao Shihu
# @Email   : shihujiao@163.com

import pickle
import torch
import argparse
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer


def read_fasta_file(filepath):
    with open(filepath, "r") as file:  
        lines = [line.strip() for line in file]
        sequences_dict = {lines[i][1:]: lines[i+1] for i in range(0, len(lines), 2)}
    return sequences_dict



def get_plm_representation(sequences_dict):
    representations = {}
    peptide_names = sequences_dict.keys()
    for name in peptide_names:
        seq = sequences_dict[name]
        seq = [' '.join(seq)]
        ids = tokenizer(seq, add_special_tokens=True, padding=True, return_tensors='pt')
        
        input_ids = ids['input_ids'].clone().detach().to('cuda')
        attention_mask = ids['attention_mask'].clone().detach().to('cuda')

        
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
        
        embedding = embedding_repr.last_hidden_state.cpu().numpy()

        
        seq_len = (attention_mask == 1).sum()
        seq_emd = embedding[0, :seq_len-1, :]
        
        representations[name] = seq_emd
        
    return representations




def nom_representation(origin_representation, outfile):
    x_max = np.load('./inputs/x_max.npy')
    x_min = np.load('./inputs/x_min.npy')
    
    x_range = x_max - x_min
    x_range[x_range == 0] = 1 
    
    normalized_representations = {}
    for name, embeddings in origin_representation.items():
        normalized_representations[name] = (embeddings - x_min) / x_range
        
    with open(outfile, 'wb') as file:
        pickle.dump(normalized_representations, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.",description="Generate peptide sequence representations using a pre-trained T5 model.")
    parser.add_argument("-i", required=True, help="Path to the FASTA file containing sequences.")
    args = parser.parse_args()
    test_sequences = read_fasta_file(args.i)

    prot_t5_path = "./prot_t5_xl_uniref50"
    tokenizer = T5Tokenizer.from_pretrained(prot_t5_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(prot_t5_path)
    model = model.eval().cuda()

    test_representation = get_plm_representation(test_sequences)
    nom_representation(test_representation, './outputs/test_plm_representation.pkl')




