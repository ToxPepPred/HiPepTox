# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 9:00
# @Author  : Jiao Shihu
# @Email   : shihujiao@163.com

import os
import math
import argparse



def create_directories():
    directories = [
        './inputs/dssp',
        './outputs/dssp',
        './outputs/pdb2rr'
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    


# In[3]:


def write_dssp(seq_id):
    ffeat = open('./outputs/dssp/{}.dssp'.format(seq_id), 'w')
    ffeat.write('res\taa\tss\tsa\ttco\tkpa\talp\tphi\tpsi\tx\ty\tz\n')  
    fi = open('./inputs/dssp/{}.dssp'.format(seq_id), 'r')
    filines = fi.readlines()
    
    cntLn = 0 ##line counter##
    residue = '#'
    for line in filines:
            if(cntLn<1):  
                    if (line[2:(2+len(residue))] == residue):  
                            cntLn+=1
                            continue
            if(cntLn>0):  
                    if (len(line)>0):   
                            ssSeq = line[16:(16+1)]  
                            aaSeq = line[13]    
                            saSeq = line[35:(35+3)]   
                            tcoSeq = line[85:(85+6)]
                            kpaSeq = line[91:(91+6)]
                            alpSeq = line[97:(97+6)]
                            phiSeq = line[103:(103+6)]
                            psiSeq = line[109:(109+6)]
                            xSeq = line[115:(115+7)]
                            ySeq = line[122:(122+7)]
                            zSeq = line[129:(129+7)]  
                            if(ssSeq.strip() == ''): 
                                    ssSeq = 'C'
                            if(line[5:10].strip() != ""):
                                    resNum = int(line[5:10])  
                                    ffeat.write(str(resNum) + '\t' + aaSeq + '\t' + ssSeq + '\t' + saSeq + '\t' + tcoSeq + '\t' + kpaSeq + '\t' + alpSeq + '\t' + phiSeq + '\t' + psiSeq + '\t' + xSeq + '\t' + ySeq + '\t' + zSeq + '\n')     

    fi.close()
    ffeat.close()





def read_fasta_file(filepath):
    with open(filepath, "r") as file:  
        lines = [line.strip() for line in file]
        sequences_dict = {lines[i][1:]: lines[i+1] for i in range(0, len(lines), 2)}
    return sequences_dict







th = float(8)

def get_distance(x1, x2, x3, y1, y2, y3):
    
    return math.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2 + (x3 - y3) ** 2)


# In[ ]:


def cal_orientation(pdbfile, outfile):
    fpdb = open(pdbfile, 'r') 
    lines = fpdb.readlines()
    
    pos = []                      
    for atomline in lines:
        if atomline[:4] == 'ATOM':
            pos.append(atomline)
    fpdb.close()
    
    
    Ca_info = {}
    Cb_info = {}
    for line in pos:
            if(line[:4] == "ATOM"):
                    res_no = 0
                    if(line[12:16].strip() == "CA"):
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            Ca = [x, y, z]
                            res_no = int(line[22:26].strip())
                            Ca_info[res_no] = Ca

                    if(line[12:16].strip() == "CB"):
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            Cb = [x, y, z]
                            res_no = int(line[22:26].strip())
                            Cb_info[res_no] = Cb
                            
    fcb = open(outfile, "w")
    for res_no in (Ca_info):
            for res_no2 in (Ca_info):
                    if(res_no2 < (res_no + 1)):
                            continue
                    if ((res_no not in Cb_info) and (res_no2 not in Cb_info)):
                            cb_cb_distance = get_distance(Ca_info[res_no][0], Ca_info[res_no][1],  Ca_info[res_no][2], Ca_info[res_no2][0], Ca_info[res_no2][1], Ca_info[res_no2][2])
                    elif((res_no in Cb_info) and (res_no2 not in Cb_info)):
                            cb_cb_distance = get_distance(Cb_info[res_no][0], Cb_info[res_no][1],  Cb_info[res_no][2], Ca_info[res_no2][0], Ca_info[res_no2][1], Ca_info[res_no2][2])

                    elif((res_no not in Cb_info) and (res_no2 in Cb_info)):
                            cb_cb_distance = get_distance(Ca_info[res_no][0], Ca_info[res_no][1],  Ca_info[res_no][2], Cb_info[res_no2][0], Cb_info[res_no2][1], Cb_info[res_no2][2])
                    else:
                            cb_cb_distance = get_distance(Cb_info[res_no][0], Cb_info[res_no][1],  Cb_info[res_no][2], Cb_info[res_no2][0], Cb_info[res_no2][1], Cb_info[res_no2][2])
                    if(cb_cb_distance > th):
                            continue

                    fcb.write(str(res_no) + ' ' + str(res_no2) + ' 0 8 ' + '%.1f'%cb_cb_distance +"\n")
    fcb.close() 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.",description="DSSP and PDB processing pipeline")
    parser.add_argument("-i", required=True, help="Path to the FASTA file containing sequences.")
    args = parser.parse_args()

    create_directories()

    test_sequences = read_fasta_file(args.i)

    for test_id in test_sequences:
        os.system("./mkdssp/mkdssp -i ./inputs/pdb/{}.pdb -o ./inputs/dssp/{}.dssp".format(test_id, test_id))
        write_dssp(test_id)

    for test_id in test_sequences:
        cal_orientation('./inputs/pdb/{}.pdb'.format(test_id), './outputs/pdb2rr/{}.pdb2rr'.format(test_id))







