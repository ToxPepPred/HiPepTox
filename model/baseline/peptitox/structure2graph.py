import numpy as np
import pandas as pd
import torch

########## Process PDB file ##########
def get_pdb_xyz(pdb_file):
    current_pos = -1000
    X = []
    current_aa = {}  # N, CA, C, O, R
    for line in pdb_file:
        if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
            if current_aa != {}:
                R_group = []
                for atom in current_aa:
                    if atom not in ["N", "CA", "C", "O"]:
                        R_group.append(current_aa[atom])
                if R_group == []:
                    R_group = [current_aa["CA"]]
                R_group = np.array(R_group).mean(0)
                X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], R_group])
                current_aa = {}
            if line[0:4].strip() != "TER":
                current_pos = int(line[22:26].strip())

        if line[0:4].strip() == "ATOM":
            atom = line[13:16].strip()
            if atom != "H":
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                current_aa[atom] = xyz
    return np.array(X)

data = pd.read_csv("./Data/merged_data.csv")
seq_list = data['sequence'].tolist()

print("Processing PDB files...")
for seq in seq_list:
    with open(f"./Data/structure_local/{seq}.pdb", "r") as f:
        X = get_pdb_xyz(f.readlines())
        torch.save(torch.tensor(X, dtype = torch.float32), f"./Data/structure_local/{seq}.tensor")
        # break
