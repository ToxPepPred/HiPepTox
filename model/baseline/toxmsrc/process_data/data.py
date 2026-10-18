from Bio import SeqIO
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def read_fasta(fname):
    with open(fname, "rU") as f:
        seq_dict = [(record.id, record.seq._data.decode())
                    for record in SeqIO.parse(f, "fasta")]
    seq_df = pd.DataFrame(data=seq_dict, columns=["Id", "Sequence"])
    return seq_df


dic = read_fasta('../Example data/example_train.fasta')
dic.to_csv('example_train.csv')

dic = read_fasta('../Example data/example_test.fasta')
dic.to_csv('example_test.csv')