# -*- coding: utf-8 -*-
# @Author  : clq
# @FileName: tools.py
# @Software: PyCharm
import os

import gensim.models
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
from Bio import SeqIO
from model import ourmodel


def read_fasta(fname):
    with open(fname, "rU") as f:
        seq_dict = [(record.id, record.seq._data.decode())
                    for record in SeqIO.parse(f, "fasta")]
    seq_df = pd.DataFrame(data=seq_dict, columns=["Id", "Sequence"])
    return seq_df


# 定义函数
def Gen_Words(sequences, kmer_len, s):
    out = []
    for i in sequences:
        kmer_list = []
        for j in range(0, (len(i) - kmer_len) + 1, s):
            kmer_list.append(i[j:j + kmer_len])
        out.append(kmer_list)
    return out


def ArgsGet():
    parse = argparse.ArgumentParser(description='ToxMSRC')
    parse.add_argument('--file', type=str, default='Raw data/test1.fasta', help='fasta file')
    parse.add_argument('--outfile', type=str, default='supple_test.fasta', help='fasta file')
    parse.add_argument('--out_path', type=str, default='result', help='output path')
    args = parse.parse_args()
    return args


def read_data(Filename):
    dictin = {}
    i = 1
    with open(Filename) as fid:
        name = None
        for line in fid:
            if line.startswith('#'):
                continue
            if line.startswith('>'):
                name = line.strip()[1:] + str(i)
                dictin[name] = ''
                i += 1
            else:
                if name is None:
                    continue
                dictin[name] += line.strip()
    return dictin


def fixOneSeq(seqIn, fixFrontScale, cutFrontScale, spcLen, paddingRes='X'):
    if len(seqIn) > spcLen:
        # cut
        exceedLen = len(seqIn) - spcLen
        frontLen = int(np.rint(float(exceedLen) * cutFrontScale))
        # lastLen = exceedLen - frontLen
        outSeq = seqIn[frontLen:frontLen + spcLen]
    elif len(seqIn) < spcLen:
        exceedLen = spcLen - len(seqIn)
        frontLen = int(np.rint(float(exceedLen) * fixFrontScale))
        lastLen = exceedLen - frontLen
        outSeq = ''
        outSeq += paddingRes * frontLen
        outSeq += seqIn
        outSeq += paddingRes * lastLen
    else:
        outSeq = seqIn
    return outSeq


def printout(fileout, Dictin):
    with open(fileout, 'w') as FIDO:
        for k in Dictin:
            FIDO.write('>%s\n' % k)
            tmpstr = Dictin[k]
            FIDO.write('%s\n' % tmpstr)


# 对代码进行整合
def supple_X(in_Filename, out_Filename, maxl):
    Dictin = read_data(in_Filename)
    outDict = {}
    for k in Dictin:
        tmpOut = fixOneSeq(Dictin[k], 0, 0, maxl, paddingRes='X')
        outDict[k] = tmpOut
    printout(out_Filename, outDict)


def process_data(file, outfile):
    supple_X(file, outfile, 50)
    seq_data = read_fasta(outfile)
    data = seq_data.iloc[:, 1].to_numpy()

    return data


def predict(model, data, output_path):
    model.load_weights('model/ToxMSRC.h5')
    y_p = model.predict([data])
    # output_file = os.path.join(output_path, 'result.txt')
    print(y_p)
    # np.savetxt(output_file, y_p[:, 1])


if __name__ == '__main__':
    args = ArgsGet()
    file = args.file
    outfile = args.outfile
    output_path = args.out_path
    # building output path directory
    Path(output_path).mkdir(exist_ok=True)

    # reading file
    data = process_data(file, outfile)
    W_model = gensim.models.Word2Vec.load('word2vec_model/word2vec.model')
    x_test3 = Gen_Words(data, 2, 1)
    X_test = []
    for i in range(0, len(x_test3)):
        s = []
        for word in x_test3[i]:
            if word in W_model.wv:
                s.append(W_model.wv[word])
            else:
                s.append(np.zeros([150, ]))
        X_test.append(s)
    X_test = np.array(X_test)
    model = ourmodel()
    predict(model, X_test, output_path)
