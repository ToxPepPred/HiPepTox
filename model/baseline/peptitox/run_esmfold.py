import os
import torch
import esm
import pandas as pd
import biotite.structure.io as bsio

if not os.path.exists("./Data/structure_local"):
    os.makedirs("./Data/structure_local")

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
# model.set_chunk_size(128)

# sequence = "GCGDINAACKSDCDCCGNSVTCDCYFTDCKCRESAIRKQF"
data = pd.read_csv("./Data/merged_data.csv")
seq_list = data['sequence'].tolist()
# Multimer prediction can be done with chains separated by ':'

for idx, sequence in enumerate(seq_list):
    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(f"./Data/structure_local/{sequence}.pdb", "w") as f:
        f.write(output)

    struct = bsio.load_structure(f"./Data/structure_local/{sequence}.pdb", extra_fields=["b_factor"])
    print(f"{idx + 1}: {sequence}: {struct.b_factor.mean()}")  # this will be the pLDDT
    # break
    # 88.3
