import csv
import json
import os
import random
from collections import defaultdict

import torch
from torch.utils import data as torch_data
from torchdrug import core
from torchdrug.core import Registry as R

from util.data import HeterogeneousPeptideGraph


CONSERVATIVE_SUBSTITUTIONS = {
    "A": ("S", "T", "V"),
    "D": ("E", "N"),
    "E": ("D", "Q"),
    "F": ("Y", "L"),
    "H": ("Q", "Y", "R"),
    "I": ("L", "V", "M"),
    "K": ("R", "H", "Q"),
    "L": ("I", "V", "M", "F"),
    "M": ("L", "I", "V"),
    "N": ("Q", "D", "S"),
    "Q": ("N", "E", "H"),
    "R": ("K", "H", "Q"),
    "S": ("T", "A", "N"),
    "T": ("S", "A", "V"),
    "V": ("I", "L", "A", "T"),
    "Y": ("F", "H"),
}


@R.register("datasets.PeptideToxicity")
class PeptideToxicityDataset(torch_data.Dataset, core.Configurable):
    target_fields = ["toxicity"]
    splits = ["train", "validation", "test"]

    def __init__(self, path, csv_name="peptide_toxicity.csv", view="residue-atom", target_fields=None,
                 lazy=False, verbose=1, augment="none", augment_prob=0.0, augment_max_mutations=1,
                 augment_seed=0, plm_embedding_file=None, plm_strict=True, plm_allow_augment_mismatch=False,
                 structure_feature_file=None, structure_feature_strict=True,
                 structure_feature_allow_augment_mismatch=False,
                 contact_edge_file=None, contact_edge_strict=False,
                 **kwargs):
        del verbose
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError("Dataset path `%s` does not exist" % path)

        self.path = path
        self.csv_name = csv_name
        self.view = view
        self.lazy = lazy
        self.kwargs = kwargs
        self.augment = (augment or "none").strip().lower()
        self.augment_prob = float(augment_prob)
        self.augment_max_mutations = max(1, int(augment_max_mutations))
        self.rng = random.Random(augment_seed)
        self.plm_embedding_file = os.path.expanduser(plm_embedding_file) if plm_embedding_file else None
        self.plm_strict = bool(plm_strict)
        self.plm_allow_augment_mismatch = bool(plm_allow_augment_mismatch)
        self.plm_embeddings = self._load_plm_embeddings(self.plm_embedding_file)
        self.structure_feature_file = os.path.expanduser(structure_feature_file) if structure_feature_file else None
        self.structure_feature_strict = bool(structure_feature_strict)
        self.structure_feature_allow_augment_mismatch = bool(structure_feature_allow_augment_mismatch)
        self.structure_features = self._load_structure_features(self.structure_feature_file)
        self.contact_edge_file = os.path.expanduser(contact_edge_file) if contact_edge_file else None
        self.contact_edge_strict = bool(contact_edge_strict)
        self.contact_edges = self._load_contact_edges(self.contact_edge_file)

        if self.augment not in {"none", "conservative_substitution"}:
            raise ValueError("Unknown augment mode `%s`" % self.augment)
        if self.plm_embeddings is not None and self.augment != "none" and not self.plm_allow_augment_mismatch:
            raise ValueError(
                "PLM embeddings are precomputed for the original sequence, so sequence augmentation must be disabled. "
                "Set `augment: none` or explicitly allow mismatch."
            )
        if self.structure_features is not None and self.augment != "none" and not self.structure_feature_allow_augment_mismatch:
            raise ValueError(
                "Structure features are precomputed for the original sequence, so sequence augmentation must be disabled. "
                "Set `augment: none` or explicitly allow mismatch."
            )

        if target_fields is not None:
            self.target_fields = target_fields

        self.csv_file = csv_name
        if not os.path.isabs(self.csv_file):
            self.csv_file = os.path.join(self.path, self.csv_file)
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError("Dataset file `%s` does not exist" % self.csv_file)

        self.load_csv(self.csv_file)

    def __len__(self):
        return len(self.sequences)

    def _normalize_split(self, split_name):
        split_name = (split_name or "train").strip().lower()
        if split_name in {"valid", "val"}:
            split_name = "validation"
        if split_name not in self.splits:
            raise ValueError("Unknown split `%s`" % split_name)
        return split_name

    def _build_graph(self, sequence, sample_id=None):
        if self.view == "residue":
            atom_mode = "none"
        elif self.view in {"atom-peptide", "residue-atom", "template-atom"}:
            atom_mode = "template"
        elif self.view in {"pseudo", "pseudo-atom"}:
            atom_mode = "pseudo"
        else:
            raise ValueError("Unknown view `%s`" % self.view)
        contact_edges = None
        if self.contact_edges is not None:
            contact_edges = self.contact_edges.get(sample_id)
            if contact_edges is None and self.contact_edge_strict:
                raise KeyError("No residue contact edges found for sample `%s`" % sample_id)
        return HeterogeneousPeptideGraph.from_sequence(
            sequence,
            contact_edges=contact_edges,
            atom_mode=atom_mode,
        )

    def _load_plm_embeddings(self, embedding_file):
        if not embedding_file:
            return None
        if not os.path.exists(embedding_file):
            raise FileNotFoundError("PLM embedding file `%s` does not exist" % embedding_file)

        payload = torch.load(embedding_file, map_location="cpu")
        if isinstance(payload, dict) and "by_sequence" in payload:
            by_sequence = payload["by_sequence"]
        elif isinstance(payload, dict):
            by_sequence = payload
        else:
            raise ValueError("Unsupported PLM embedding format in `%s`" % embedding_file)

        normalized = {}
        for sequence, embedding in by_sequence.items():
            normalized_sequence = HeterogeneousPeptideGraph.normalize_sequence(sequence)
            normalized[normalized_sequence] = torch.as_tensor(embedding, dtype=torch.float)
        return normalized

    def _load_contact_edges(self, contact_edge_file):
        if not contact_edge_file:
            return None
        if not os.path.exists(contact_edge_file):
            raise FileNotFoundError("Contact edge file `%s` does not exist" % contact_edge_file)

        if contact_edge_file.endswith(".json"):
            with open(contact_edge_file, "r", encoding="utf-8") as fin:
                payload = json.load(fin)
        else:
            payload = torch.load(contact_edge_file, map_location="cpu")

        if isinstance(payload, dict) and "by_id" in payload:
            payload = payload["by_id"]
        if not isinstance(payload, dict):
            raise ValueError("Unsupported contact edge format in `%s`" % contact_edge_file)

        normalized = {}
        for sample_id, edges in payload.items():
            normalized[str(sample_id)] = [tuple(int(index) for index in edge) for edge in edges]
        return normalized

    def _attach_plm_embedding(self, graph, sequence):
        if self.plm_embeddings is None:
            return graph

        normalized_sequence = HeterogeneousPeptideGraph.normalize_sequence(sequence)
        embedding = self.plm_embeddings.get(normalized_sequence)
        if embedding is None:
            if self.plm_strict:
                raise KeyError("No PLM embedding found for sequence `%s`" % normalized_sequence)
            return graph

        return graph.append_residue_embedding(embedding)

    def _load_structure_features(self, structure_feature_file):
        if not structure_feature_file:
            return None
        if not os.path.exists(structure_feature_file):
            raise FileNotFoundError("Structure feature file `%s` does not exist" % structure_feature_file)

        if structure_feature_file.endswith(".json"):
            with open(structure_feature_file, "r", encoding="utf-8") as fin:
                payload = json.load(fin)
        else:
            payload = torch.load(structure_feature_file, map_location="cpu")

        if isinstance(payload, dict) and "by_id" in payload:
            payload = payload["by_id"]
        if not isinstance(payload, dict):
            raise ValueError("Unsupported structure feature format in `%s`" % structure_feature_file)

        normalized = {}
        for sample_id, feature in payload.items():
            normalized[str(sample_id)] = torch.as_tensor(feature, dtype=torch.float)
        return normalized

    def _attach_structure_feature(self, graph, sample_id):
        if self.structure_features is None:
            return graph

        feature = self.structure_features.get(sample_id)
        if feature is None:
            if self.structure_feature_strict:
                raise KeyError("No structure feature found for sample `%s`" % sample_id)
            return graph

        return graph.append_residue_embedding(feature)

    def load_csv(self, csv_file):
        self.ids = []
        self.sequences = []
        self.sources = []
        self.sample_splits = []
        self.targets = defaultdict(list)
        self.data = []
        seen_ids = set()

        with open(csv_file, "r", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            fields = set(reader.fieldnames or [])

            label_column = None
            for target_field in self.target_fields:
                if target_field in fields:
                    label_column = target_field
                    break
            if label_column is None and "label" in fields and self.target_fields == ["toxicity"]:
                label_column = "label"
            if label_column is None:
                raise ValueError("No toxicity label column found in `%s`" % csv_file)
            if "sequence" not in fields:
                raise ValueError("Column `sequence` is required in `%s`" % csv_file)

            for row in reader:
                sequence = (row.get("sequence") or "").strip().upper()
                if not sequence:
                    continue

                label = int(row[label_column])
                split_name = self._normalize_split(row.get("split"))
                sample_id = (
                    row.get("id")
                    or row.get("record_id")
                    or "sample_%d" % len(self.ids)
                ).strip()
                if not sample_id:
                    raise ValueError("Sample IDs must not be empty in `%s`" % csv_file)
                if sample_id in seen_ids:
                    raise ValueError("Duplicate sample ID `%s` in `%s`; each CSV row requires a unique ID" % (sample_id, csv_file))
                seen_ids.add(sample_id)
                graph = None
                if not self.lazy:
                    graph = self._build_graph(sequence, sample_id=sample_id)
                    graph = self._attach_structure_feature(graph, sample_id)
                    graph = self._attach_plm_embedding(graph, sequence)

                self.ids.append(sample_id)
                self.sequences.append(sequence)
                self.sources.append((row.get("source") or "").strip())
                self.sample_splits.append(split_name)
                self.targets["toxicity"].append(label)
                self.data.append(graph)

    def split(self):
        train_indices = [i for i, split_name in enumerate(self.sample_splits) if split_name == "train"]
        valid_indices = [i for i, split_name in enumerate(self.sample_splits) if split_name == "validation"]
        test_indices = [i for i, split_name in enumerate(self.sample_splits) if split_name == "test"]

        return [
            torch_data.Subset(self, train_indices),
            torch_data.Subset(self, valid_indices),
            torch_data.Subset(self, test_indices),
        ]

    def _should_augment(self, index):
        return (
            self.augment != "none"
            and self.sample_splits[index] == "train"
            and self.augment_prob > 0
            and self.rng.random() < self.augment_prob
        )

    def _augment_sequence(self, sequence):
        if self.augment == "none":
            return sequence
        if self.augment != "conservative_substitution":
            raise ValueError("Unknown augment mode `%s`" % self.augment)

        candidate_positions = [
            i for i, residue in enumerate(sequence) if residue in CONSERVATIVE_SUBSTITUTIONS
        ]
        if not candidate_positions:
            return sequence

        mutation_count = min(self.augment_max_mutations, len(candidate_positions))
        mutation_count = self.rng.randint(1, mutation_count)
        chosen_positions = self.rng.sample(candidate_positions, mutation_count)
        residues = list(sequence)
        for position in chosen_positions:
            choices = CONSERVATIVE_SUBSTITUTIONS[residues[position]]
            residues[position] = self.rng.choice(choices)
        return "".join(residues)

    def get_item(self, index):
        sequence = self.sequences[index]
        graph = self.data[index]

        if self._should_augment(index):
            augmented_sequence = self._augment_sequence(sequence)
            graph = self._build_graph(augmented_sequence, sample_id=self.ids[index])
            if self.structure_features is not None:
                graph = self._attach_structure_feature(graph, self.ids[index])
            if self.plm_embeddings is not None:
                graph = self._attach_plm_embedding(graph, sequence)
            sequence = augmented_sequence
        else:
            if self.lazy and graph is None:
                graph = self._build_graph(sequence, sample_id=self.ids[index])
                graph = self._attach_structure_feature(graph, self.ids[index])
                graph = self._attach_plm_embedding(graph, sequence)
                self.data[index] = graph

        item = {
            "graph": graph,
            "id": self.ids[index],
            "sequence": sequence,
            "source": self.sources[index],
        }
        target = {field: self.targets[field][index] for field in self.target_fields}
        item.update(target)
        return item

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)

        index = self._standarize_index(index, len(self))
        return [self.get_item(i) for i in index]

    def _standarize_index(self, index, count):
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = range(start, stop, step)
        elif not isinstance(index, list):
            raise ValueError("Unknown index `%s`" % index)
        return index
