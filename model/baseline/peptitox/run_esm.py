import os
import torch
import esm
import pandas as pd

data = pd.read_csv("./Data/merged_data.csv")
seq_list = data['sequence'].tolist()
save_dir = "./Data/esm_embedding"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval().cuda()  # disables dropout for deterministic results

repr_layers = 33  # layers to extract

for idx, seq in enumerate(seq_list):
    batch_data = [("peptide1", seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[repr_layers], return_contacts=False)
    token_representations = results["representations"][repr_layers]
    peptide_embedding = token_representations[0, 1: batch_lens[0] - 1]
    torch.save(peptide_embedding, os.path.join(save_dir, f"{seq}.pt"))
    print(f"Sequence {idx + 1}: {seq} saved successfully.")
    # peptide_embedding = token_representations[i, 1: tokens_len - 1]

