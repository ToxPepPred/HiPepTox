import argparse
import csv
import os
from pathlib import Path

import torch

RESIDUE_TYPES = set("ARNDCQEGHILKMFPSTWYVX")


def normalize_sequence(sequence):
    sequence = (sequence or "").strip().upper()
    return "".join(token if token in RESIDUE_TYPES else "X" for token in sequence)


def read_unique_sequences(csv_paths):
    unique = []
    seen = set()
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            if "sequence" not in (reader.fieldnames or []):
                raise ValueError("Column `sequence` is required in `%s`" % csv_path)
            for row in reader:
                sequence = normalize_sequence(row.get("sequence"))
                if not sequence or sequence in seen:
                    continue
                seen.add(sequence)
                unique.append(sequence)
    return unique


def load_esm_model(model_name_or_path, device):
    try:
        import esm
    except ImportError as exc:
        raise ImportError("`esm` is required for ESM2 extraction in the current environment") from exc

    if os.path.isfile(model_name_or_path):
        model_path = Path(model_name_or_path)
        original_torch_load = torch.load

        # fair-esm local checkpoints were created before PyTorch 2.6 changed
        # torch.load(..., weights_only=True) to the default. These checkpoints
        # include metadata such as argparse.Namespace, so we temporarily force
        # the legacy behavior when loading trusted local model files.
        def torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_compat
        try:
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(model_path))
        finally:
            torch.load = original_torch_load
    else:
        if not hasattr(esm.pretrained, model_name_or_path):
            raise ValueError("Unknown ESM model `%s`" % model_name_or_path)
        model, alphabet = getattr(esm.pretrained, model_name_or_path)()

    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def pool_representations(representations, repr_layers, layer_pool):
    tensors = [representations[layer] for layer in repr_layers]
    if layer_pool == "mean":
        return torch.stack(tensors, dim=0).mean(dim=0)
    raise ValueError("Unsupported layer pool `%s`" % layer_pool)


def extract_esm_embeddings(model, batch_converter, sequences, batch_size, repr_layers, layer_pool, device):
    embeddings = {}
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start:start + batch_size]
        batch = [("seq_%d" % (start + i), sequence) for i, sequence in enumerate(batch_sequences)]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)

        with torch.no_grad():
            output = model(tokens, repr_layers=repr_layers, return_contacts=False)
            representation = pool_representations(output["representations"], repr_layers, layer_pool).cpu()

        for index, sequence in enumerate(batch_sequences):
            embeddings[sequence] = representation[index, 1: len(sequence) + 1].clone()
    return embeddings


def load_prott5_model(model_name_or_path, device):
    try:
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError as exc:
        raise ImportError(
            "`transformers` and `sentencepiece` are required for ProtT5 extraction. "
            "They are not installed in the current environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name_or_path).eval().to(device)
    return model, tokenizer


def extract_prott5_embeddings(model, tokenizer, sequences, batch_size, device):
    embeddings = {}
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start:start + batch_size]
        encoded_sequences = [" ".join(list(sequence)) for sequence in batch_sequences]
        batch = tokenizer(
            encoded_sequences,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            hidden = model(**batch).last_hidden_state.cpu()

        attention_mask = batch["attention_mask"].cpu()
        for index, sequence in enumerate(batch_sequences):
            token_count = int(attention_mask[index].sum().item())
            embeddings[sequence] = hidden[index, :token_count - 1].clone()
    return embeddings


def save_embeddings(output_file, backend, model_name_or_path, embedding_dict, repr_layers, layer_pool):
    first_embedding = next(iter(embedding_dict.values()))
    payload = {
        "backend": backend,
        "model_name_or_path": model_name_or_path,
        "repr_layer": repr_layers[0] if len(repr_layers) == 1 else None,
        "repr_layers": repr_layers,
        "layer_pool": layer_pool,
        "embedding_dim": int(first_embedding.shape[-1]),
        "by_sequence": embedding_dict,
    }
    torch.save(payload, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csvs", nargs="+", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--backend", choices=["esm", "prot_t5"], default="esm")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--repr_layer", type=int, default=None)
    parser.add_argument("--repr_layers", type=int, nargs="+", default=None)
    parser.add_argument("--layer_pool", choices=["mean"], default="mean")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sequences = read_unique_sequences(args.input_csvs)
    if not sequences:
        raise ValueError("No sequences found in the provided CSV files")

    device = torch.device(args.device)
    if args.backend == "esm":
        model, batch_converter = load_esm_model(args.model_name_or_path, device)
        if args.repr_layers:
            repr_layers = sorted(set(args.repr_layers))
        else:
            repr_layers = [args.repr_layer or getattr(model, "num_layers")]
        embeddings = extract_esm_embeddings(
            model, batch_converter, sequences, args.batch_size, repr_layers, args.layer_pool, device
        )
    else:
        model, tokenizer = load_prott5_model(args.model_name_or_path, device)
        repr_layers = []
        embeddings = extract_prott5_embeddings(model, tokenizer, sequences, args.batch_size, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    save_embeddings(args.output_file, args.backend, args.model_name_or_path, embeddings, repr_layers, args.layer_pool)
    print(
        "Wrote %d sequence embeddings to %s"
        % (len(embeddings), os.path.abspath(args.output_file))
    )


if __name__ == "__main__":
    main()
