from collections import defaultdict

import torch
import torch.nn.functional as F

from torchdrug.data import Graph, PackedGraph


def _ordered_unique(items):
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


BACKBONE_ATOMS = ["N", "CA", "C", "O"]
BACKBONE_ELEMENTS = {"N": "N", "CA": "C", "C": "C", "O": "O"}
BACKBONE_BONDS = [
    ("N", "CA", "single"),
    ("CA", "C", "single"),
    ("C", "O", "double"),
]


RESIDUE_TEMPLATES = {
    "A": {
        "side_atoms": ["CB"],
        "elements": {"CB": "C"},
        "bonds": [("CA", "CB", "single")],
    },
    "R": {
        "side_atoms": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "elements": {"CB": "C", "CG": "C", "CD": "C", "NE": "N", "CZ": "C", "NH1": "N", "NH2": "N"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "CD", "single"),
            ("CD", "NE", "single"),
            ("NE", "CZ", "single"),
            ("CZ", "NH1", "double"),
            ("CZ", "NH2", "single"),
        ],
    },
    "N": {
        "side_atoms": ["CB", "CG", "OD1", "ND2"],
        "elements": {"CB": "C", "CG": "C", "OD1": "O", "ND2": "N"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "OD1", "double"),
            ("CG", "ND2", "single"),
        ],
    },
    "D": {
        "side_atoms": ["CB", "CG", "OD1", "OD2"],
        "elements": {"CB": "C", "CG": "C", "OD1": "O", "OD2": "O"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "OD1", "double"),
            ("CG", "OD2", "single"),
        ],
    },
    "C": {
        "side_atoms": ["CB", "SG"],
        "elements": {"CB": "C", "SG": "S"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "SG", "single"),
        ],
    },
    "Q": {
        "side_atoms": ["CB", "CG", "CD", "OE1", "NE2"],
        "elements": {"CB": "C", "CG": "C", "CD": "C", "OE1": "O", "NE2": "N"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "CD", "single"),
            ("CD", "OE1", "double"),
            ("CD", "NE2", "single"),
        ],
    },
    "E": {
        "side_atoms": ["CB", "CG", "CD", "OE1", "OE2"],
        "elements": {"CB": "C", "CG": "C", "CD": "C", "OE1": "O", "OE2": "O"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "CD", "single"),
            ("CD", "OE1", "double"),
            ("CD", "OE2", "single"),
        ],
    },
    "G": {
        "side_atoms": [],
        "elements": {},
        "bonds": [],
    },
    "H": {
        "side_atoms": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "elements": {"CB": "C", "CG": "C", "ND1": "N", "CD2": "C", "CE1": "C", "NE2": "N"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "ND1", "aromatic"),
            ("ND1", "CE1", "aromatic"),
            ("CE1", "NE2", "aromatic"),
            ("NE2", "CD2", "aromatic"),
            ("CD2", "CG", "aromatic"),
        ],
    },
    "I": {
        "side_atoms": ["CB", "CG1", "CG2", "CD1"],
        "elements": {"CB": "C", "CG1": "C", "CG2": "C", "CD1": "C"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG1", "single"),
            ("CB", "CG2", "single"),
            ("CG1", "CD1", "single"),
        ],
    },
    "L": {
        "side_atoms": ["CB", "CG", "CD1", "CD2"],
        "elements": {"CB": "C", "CG": "C", "CD1": "C", "CD2": "C"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "CD1", "single"),
            ("CG", "CD2", "single"),
        ],
    },
    "K": {
        "side_atoms": ["CB", "CG", "CD", "CE", "NZ"],
        "elements": {"CB": "C", "CG": "C", "CD": "C", "CE": "C", "NZ": "N"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "CD", "single"),
            ("CD", "CE", "single"),
            ("CE", "NZ", "single"),
        ],
    },
    "M": {
        "side_atoms": ["CB", "CG", "SD", "CE"],
        "elements": {"CB": "C", "CG": "C", "SD": "S", "CE": "C"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "SD", "single"),
            ("SD", "CE", "single"),
        ],
    },
    "F": {
        "side_atoms": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "elements": {"CB": "C", "CG": "C", "CD1": "C", "CD2": "C", "CE1": "C", "CE2": "C", "CZ": "C"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "CD1", "aromatic"),
            ("CG", "CD2", "aromatic"),
            ("CD1", "CE1", "aromatic"),
            ("CD2", "CE2", "aromatic"),
            ("CE1", "CZ", "aromatic"),
            ("CE2", "CZ", "aromatic"),
        ],
    },
    "P": {
        "side_atoms": ["CB", "CG", "CD"],
        "elements": {"CB": "C", "CG": "C", "CD": "C"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "CD", "single"),
            ("CD", "N", "single"),
        ],
    },
    "S": {
        "side_atoms": ["CB", "OG"],
        "elements": {"CB": "C", "OG": "O"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "OG", "single"),
        ],
    },
    "T": {
        "side_atoms": ["CB", "OG1", "CG2"],
        "elements": {"CB": "C", "OG1": "O", "CG2": "C"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "OG1", "single"),
            ("CB", "CG2", "single"),
        ],
    },
    "W": {
        "side_atoms": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "elements": {
            "CB": "C",
            "CG": "C",
            "CD1": "C",
            "CD2": "C",
            "NE1": "N",
            "CE2": "C",
            "CE3": "C",
            "CZ2": "C",
            "CZ3": "C",
            "CH2": "C",
        },
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "CD1", "aromatic"),
            ("CG", "CD2", "aromatic"),
            ("CD1", "NE1", "aromatic"),
            ("NE1", "CE2", "aromatic"),
            ("CE2", "CD2", "aromatic"),
            ("CD2", "CE3", "aromatic"),
            ("CE3", "CZ3", "aromatic"),
            ("CZ3", "CH2", "aromatic"),
            ("CH2", "CZ2", "aromatic"),
            ("CZ2", "CE2", "aromatic"),
        ],
    },
    "Y": {
        "side_atoms": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        "elements": {"CB": "C", "CG": "C", "CD1": "C", "CD2": "C", "CE1": "C", "CE2": "C", "CZ": "C", "OH": "O"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG", "single"),
            ("CG", "CD1", "aromatic"),
            ("CG", "CD2", "aromatic"),
            ("CD1", "CE1", "aromatic"),
            ("CD2", "CE2", "aromatic"),
            ("CE1", "CZ", "aromatic"),
            ("CE2", "CZ", "aromatic"),
            ("CZ", "OH", "single"),
        ],
    },
    "V": {
        "side_atoms": ["CB", "CG1", "CG2"],
        "elements": {"CB": "C", "CG1": "C", "CG2": "C"},
        "bonds": [
            ("CA", "CB", "single"),
            ("CB", "CG1", "single"),
            ("CB", "CG2", "single"),
        ],
    },
    "X": {
        "side_atoms": [],
        "elements": {},
        "bonds": [],
    },
}


RESIDUE_TYPES = list("ARNDCQEGHILKMFPSTWYV") + ["X"]
RESIDUE2ID = dict(zip(RESIDUE_TYPES, range(len(RESIDUE_TYPES))))
ID2RESIDUE = dict(enumerate(RESIDUE_TYPES))

ATOM_ELEMENT_TYPES = ["C", "N", "O", "S", "X"]
ATOM_ELEMENT2ID = dict(zip(ATOM_ELEMENT_TYPES, range(len(ATOM_ELEMENT_TYPES))))
ID2ATOM_ELEMENT = dict(enumerate(ATOM_ELEMENT_TYPES))

ATOM_NAME_TYPES = _ordered_unique(
    BACKBONE_ATOMS +
    [atom_name for template in RESIDUE_TEMPLATES.values() for atom_name in template["side_atoms"]] +
    ["UNK"]
)
ATOM_NAME2ID = dict(zip(ATOM_NAME_TYPES, range(len(ATOM_NAME_TYPES))))
ID2ATOM_NAME = dict(enumerate(ATOM_NAME_TYPES))

ATOM_OFFSET = len(RESIDUE_TYPES)

HYBRIDIZATION_TYPES = ["other", "sp", "sp2", "sp3"]
HYBRIDIZATION2ID = dict(zip(HYBRIDIZATION_TYPES, range(len(HYBRIDIZATION_TYPES))))
ID2HYBRIDIZATION = dict(enumerate(HYBRIDIZATION_TYPES))

CHIRALITY_TYPES = ["achiral", "chiral", "unknown"]
CHIRALITY2ID = dict(zip(CHIRALITY_TYPES, range(len(CHIRALITY_TYPES))))
ID2CHIRALITY = dict(enumerate(CHIRALITY_TYPES))

BOND_ORDER = {
    "single": 1.0,
    "double": 2.0,
    "aromatic": 1.5,
    "peptide": 1.0,
}

FORMAL_CHARGE = {
    "K": {"NZ": 1.0},
    "R": {"NH1": 1.0},
    "D": {"OD2": -1.0},
    "E": {"OE2": -1.0},
}

TYPICAL_COORDINATION = {
    "C": {"sp": 2.0, "sp2": 3.0, "sp3": 4.0, "other": 4.0},
    "N": {"sp": 2.0, "sp2": 3.0, "sp3": 3.0, "other": 3.0},
    "O": {"sp": 1.0, "sp2": 1.0, "sp3": 2.0, "other": 2.0},
    "S": {"sp": 2.0, "sp2": 2.0, "sp3": 2.0, "other": 2.0},
    "X": {"sp": 0.0, "sp2": 0.0, "sp3": 0.0, "other": 0.0},
}

RELATION_TYPES = [
    "residue_backbone",
    "peptide_bond",
    "atom_bond_single",
    "atom_bond_double",
    "atom_bond_aromatic",
    "atom_to_residue",
    "residue_to_atom",
    "residue_contact",
]
RELATION2ID = dict(zip(RELATION_TYPES, range(len(RELATION_TYPES))))
ID2RELATION = dict(enumerate(RELATION_TYPES))


class HeterogeneousPeptideGraph(Graph):
    """
    Hierarchical peptide graph with residue nodes and template-expanded heavy atoms.

    Each residue is represented twice:
    - one residue node for sequence-level message passing
    - a set of heavy-atom nodes constructed from standard amino-acid templates

    This is the template-based analogue of GlycanAA style hierarchical graphs.
    """

    _meta_types = {"node", "edge", "graph", "node reference", "edge reference", "graph reference"}

    residue_types = RESIDUE_TYPES
    residue2id = RESIDUE2ID
    id2residue = ID2RESIDUE

    atom_element_types = ATOM_ELEMENT_TYPES
    atom_element2id = ATOM_ELEMENT2ID
    id2atom_element = ID2ATOM_ELEMENT

    atom_name_types = ATOM_NAME_TYPES
    atom_name2id = ATOM_NAME2ID
    id2atom_name = ID2ATOM_NAME
    hybridization_types = HYBRIDIZATION_TYPES
    hybridization2id = HYBRIDIZATION2ID
    id2hybridization = ID2HYBRIDIZATION
    chirality_types = CHIRALITY_TYPES
    chirality2id = CHIRALITY2ID
    id2chirality = ID2CHIRALITY

    atom_offset = ATOM_OFFSET
    relation_types = RELATION_TYPES
    relation2id = RELATION2ID
    id2relation = ID2RELATION

    packed_type = None

    def __init__(self, edge_list=None, unit_type=None, link_type=None, unit_feature=None, link_feature=None,
                 cumsum=None, sequence=None, **kwargs):
        if "num_relation" not in kwargs:
            kwargs["num_relation"] = len(self.relation_types)
        super().__init__(edge_list=edge_list, **kwargs)
        unit_type, link_type = self._standarize_unit_link(unit_type, link_type)

        self.cumsum = cumsum or []
        with self.unit():
            self.unit_type = unit_type
            if unit_feature is not None:
                self.unit_feature = torch.as_tensor(unit_feature, device=self.device)

        with self.link():
            self.link_type = link_type
            if link_feature is not None:
                self.link_feature = torch.as_tensor(link_feature, device=self.device)

    def _standarize_unit_link(self, unit_type, link_type):
        if unit_type is None:
            raise ValueError("`unit_type` should be provided")
        if link_type is None:
            raise ValueError("`link_type` should be provided")

        unit_type = torch.as_tensor(unit_type, dtype=torch.long, device=self.device)
        link_type = torch.as_tensor(link_type, dtype=torch.long, device=self.device)
        return unit_type, link_type

    @classmethod
    def normalize_sequence(cls, sequence):
        sequence = (sequence or "").strip().upper()
        return "".join(token if token in cls.residue2id else "X" for token in sequence)

    @classmethod
    def num_unit_type(cls):
        return len(cls.residue_types) + len(cls.atom_element_types)

    @classmethod
    def node_feature_dim(cls):
        return (
            len(cls.residue_types) +
            len(cls.atom_element_types) +
            len(cls.atom_name_types) +
            len(cls.residue_types) +
            4 + 2 + 1 +
            6 +
            len(cls.hybridization_types) +
            len(cls.chirality_types)
        )

    @classmethod
    def _empty_atom_descriptor_feature(cls):
        return torch.zeros(6 + len(cls.hybridization_types) + len(cls.chirality_types), dtype=torch.float)

    @classmethod
    def _atom_unit_type(cls, element):
        element = element if element in cls.atom_element2id else "X"
        return cls.atom_offset + cls.atom_element2id[element]

    @classmethod
    def _relation_from_bond_type(cls, bond_type):
        mapping = {
            "single": "atom_bond_single",
            "double": "atom_bond_double",
            "aromatic": "atom_bond_aromatic",
            "peptide": "peptide_bond",
        }
        if bond_type not in mapping:
            raise ValueError("Unknown bond type `%s`" % bond_type)
        return cls.relation2id[mapping[bond_type]]

    @classmethod
    def _bond_order(cls, bond_type):
        if bond_type not in BOND_ORDER:
            raise ValueError("Unknown bond type `%s`" % bond_type)
        return BOND_ORDER[bond_type]

    @classmethod
    def _ring_atoms(cls, atom_names, bonds):
        adjacency = {atom_name: set() for atom_name in atom_names}
        for src, dst, _ in bonds:
            if src not in adjacency or dst not in adjacency:
                continue
            adjacency[src].add(dst)
            adjacency[dst].add(src)

        pruned = {atom_name: set(neighbors) for atom_name, neighbors in adjacency.items()}
        queue = [atom_name for atom_name, neighbors in pruned.items() if len(neighbors) <= 1]
        removed = set()

        while queue:
            atom_name = queue.pop()
            if atom_name in removed:
                continue
            removed.add(atom_name)
            for neighbor in list(pruned[atom_name]):
                pruned[neighbor].discard(atom_name)
                if len(pruned[neighbor]) <= 1 and neighbor not in removed:
                    queue.append(neighbor)
            pruned[atom_name].clear()

        return {atom_name for atom_name in atom_names if atom_name not in removed}

    @classmethod
    def _formal_charge(cls, residue, atom_name, residue_index):
        formal_charge = FORMAL_CHARGE.get(residue, {}).get(atom_name, 0.0)
        if residue_index == 0 and atom_name == "N":
            formal_charge += 1.0
        return formal_charge

    @classmethod
    def _hybridization(cls, element, atom_name, aromatic, bond_types, valence):
        if aromatic:
            return "sp2"

        max_order = max((cls._bond_order(bond_type) for bond_type in bond_types), default=0.0)
        if element == "C":
            if max_order >= 2.0:
                return "sp2"
            if atom_name == "C":
                return "sp2"
            return "sp3"
        if element == "N":
            if max_order >= 2.0 or atom_name in {"ND1", "NE1", "NE2"}:
                return "sp2"
            return "sp3"
        if element == "O":
            return "sp2" if max_order >= 2.0 else "sp3"
        if element == "S":
            return "sp3" if valence <= 2.0 else "sp2"
        return "other"

    @classmethod
    def _chirality(cls, residue, atom_name):
        if atom_name == "CA":
            if residue == "X":
                return "unknown"
            return "achiral" if residue == "G" else "chiral"
        if residue in {"I", "T"} and atom_name == "CB":
            return "chiral"
        return "achiral"

    @classmethod
    def _num_hydrogens(cls, element, hybridization, degree, formal_charge, atom_name, residue_index, num_residue):
        expected_coordination = TYPICAL_COORDINATION.get(element, TYPICAL_COORDINATION["X"]).get(hybridization, 0.0)
        if element == "N" and formal_charge > 0:
            expected_coordination += formal_charge
        if formal_charge < 0:
            expected_coordination += formal_charge

        effective_degree = degree
        if atom_name == "C" and residue_index == num_residue - 1:
            # The terminal carboxylate oxygen is omitted from the template graph.
            effective_degree += 1.0

        return max(expected_coordination - effective_degree, 0.0)

    @classmethod
    def _atom_descriptor_feature(cls, residue, atom_name, element, residue_index, num_residue, degree, valence,
                                 aromatic, ring, bond_types):
        formal_charge = cls._formal_charge(residue, atom_name, residue_index)
        hybridization = cls._hybridization(element, atom_name, aromatic, bond_types, valence)
        chirality = cls._chirality(residue, atom_name)
        num_hydrogens = cls._num_hydrogens(
            element, hybridization, degree, formal_charge, atom_name, residue_index, num_residue
        )

        scalar_feature = torch.tensor(
            [formal_charge, degree, valence, float(aromatic), float(ring), num_hydrogens],
            dtype=torch.float,
        )
        hybridization_feature = F.one_hot(
            torch.tensor(cls.hybridization2id[hybridization], dtype=torch.long),
            num_classes=len(cls.hybridization_types),
        ).float()
        chirality_feature = F.one_hot(
            torch.tensor(cls.chirality2id[chirality], dtype=torch.long),
            num_classes=len(cls.chirality_types),
        ).float()
        return torch.cat([scalar_feature, hybridization_feature, chirality_feature])

    @classmethod
    def _atom_descriptors(cls, residue, residue_index, num_residue, atom_names, element_lookup, bonds):
        neighbors = defaultdict(list)
        for atom_name in atom_names:
            neighbors[atom_name] = []

        for src, dst, bond_type in bonds:
            neighbors[src].append((dst, bond_type))
            neighbors[dst].append((src, bond_type))

        if residue_index > 0:
            neighbors["N"].append(("_prev_C", "peptide"))
        if residue_index < num_residue - 1:
            neighbors["C"].append(("_next_N", "peptide"))

        ring_atoms = cls._ring_atoms(atom_names, bonds)
        descriptors = {}
        for atom_name in atom_names:
            bond_types = [bond_type for _, bond_type in neighbors[atom_name]]
            degree = float(len(bond_types))
            valence = float(sum(cls._bond_order(bond_type) for bond_type in bond_types))
            aromatic = any(bond_type == "aromatic" for bond_type in bond_types)
            ring = atom_name in ring_atoms
            descriptors[atom_name] = cls._atom_descriptor_feature(
                residue=residue,
                atom_name=atom_name,
                element=element_lookup.get(atom_name, "X"),
                residue_index=residue_index,
                num_residue=num_residue,
                degree=degree,
                valence=valence,
                aromatic=aromatic,
                ring=ring,
                bond_types=bond_types,
            )

        return descriptors

    @classmethod
    def _residue_feature(cls, residue, residue_index, num_residue):
        residue_vec = F.one_hot(
            torch.tensor(cls.residue2id.get(residue, cls.residue2id["X"]), dtype=torch.long),
            num_classes=len(cls.residue_types),
        ).float()
        atom_element_vec = torch.zeros(len(cls.atom_element_types), dtype=torch.float)
        atom_name_vec = torch.zeros(len(cls.atom_name_types), dtype=torch.float)
        parent_residue_vec = residue_vec.clone()
        flags = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float)
        terminals = torch.tensor(
            [1.0 if residue_index == 0 else 0.0, 1.0 if residue_index == num_residue - 1 else 0.0],
            dtype=torch.float,
        )
        normalized_pos = torch.tensor(
            [0.0 if num_residue == 1 else residue_index / float(num_residue - 1)],
            dtype=torch.float,
        )

        return torch.cat(
            [
                residue_vec,
                atom_element_vec,
                atom_name_vec,
                parent_residue_vec,
                flags,
                terminals,
                normalized_pos,
                cls._empty_atom_descriptor_feature(),
            ]
        )

    @classmethod
    def _atom_feature(cls, element, atom_name, parent_residue, residue_index, num_residue, atom_descriptor_feature):
        residue_vec = torch.zeros(len(cls.residue_types), dtype=torch.float)
        atom_element_vec = F.one_hot(
            torch.tensor(cls.atom_element2id.get(element, cls.atom_element2id["X"]), dtype=torch.long),
            num_classes=len(cls.atom_element_types),
        ).float()
        atom_name_vec = F.one_hot(
            torch.tensor(cls.atom_name2id.get(atom_name, cls.atom_name2id["UNK"]), dtype=torch.long),
            num_classes=len(cls.atom_name_types),
        ).float()
        parent_residue_vec = F.one_hot(
            torch.tensor(cls.residue2id.get(parent_residue, cls.residue2id["X"]), dtype=torch.long),
            num_classes=len(cls.residue_types),
        ).float()
        is_backbone = 1.0 if atom_name in BACKBONE_ATOMS else 0.0
        flags = torch.tensor([0.0, 1.0, is_backbone, 1.0 - is_backbone], dtype=torch.float)
        terminals = torch.tensor(
            [1.0 if residue_index == 0 else 0.0, 1.0 if residue_index == num_residue - 1 else 0.0],
            dtype=torch.float,
        )
        normalized_pos = torch.tensor(
            [0.0 if num_residue == 1 else residue_index / float(num_residue - 1)],
            dtype=torch.float,
        )

        return torch.cat(
            [
                residue_vec,
                atom_element_vec,
                atom_name_vec,
                parent_residue_vec,
                flags,
                terminals,
                normalized_pos,
                atom_descriptor_feature,
            ]
        )

    @classmethod
    def _build_template_graph(cls, sequence, contact_edges=None):
        num_residue = len(sequence)
        unit_type = [cls.residue2id[token] for token in sequence]
        node_features = [cls._residue_feature(token, idx, num_residue) for idx, token in enumerate(sequence)]
        edge_list = []
        link_type = []

        for residue_index in range(num_residue - 1):
            src = residue_index
            dst = residue_index + 1
            relation = cls.relation2id["residue_backbone"]
            edge_list.extend([[src, dst, relation], [dst, src, relation]])
            link_type.extend([relation, relation])

        cumsum = [num_residue]
        offset = num_residue
        previous_atom_ids = None

        for residue_index, residue in enumerate(sequence):
            template = RESIDUE_TEMPLATES.get(residue, RESIDUE_TEMPLATES["X"])
            atom_ids = {}
            atom_names = BACKBONE_ATOMS + template["side_atoms"]
            element_lookup = {
                atom_name: template["elements"].get(atom_name, BACKBONE_ELEMENTS.get(atom_name, "X"))
                for atom_name in atom_names
            }
            atom_descriptors = cls._atom_descriptors(
                residue=residue,
                residue_index=residue_index,
                num_residue=num_residue,
                atom_names=atom_names,
                element_lookup=element_lookup,
                bonds=BACKBONE_BONDS + template["bonds"],
            )

            for atom_name in atom_names:
                element = element_lookup[atom_name]
                atom_ids[atom_name] = offset
                unit_type.append(cls._atom_unit_type(element))
                node_features.append(
                    cls._atom_feature(
                        element,
                        atom_name,
                        residue,
                        residue_index,
                        num_residue,
                        atom_descriptors[atom_name],
                    )
                )
                offset += 1

            for atom_src, atom_dst, bond_type in BACKBONE_BONDS + template["bonds"]:
                relation = cls._relation_from_bond_type(bond_type)
                src = atom_ids[atom_src]
                dst = atom_ids[atom_dst]
                edge_list.extend([[src, dst, relation], [dst, src, relation]])
                link_type.extend([relation, relation])

            if previous_atom_ids is not None:
                relation = cls._relation_from_bond_type("peptide")
                src = previous_atom_ids["C"]
                dst = atom_ids["N"]
                edge_list.extend([[src, dst, relation], [dst, src, relation]])
                link_type.extend([relation, relation])

            residue_to_atom = cls.relation2id["residue_to_atom"]
            atom_to_residue = cls.relation2id["atom_to_residue"]
            for atom_id in atom_ids.values():
                edge_list.append([residue_index, atom_id, residue_to_atom])
                edge_list.append([atom_id, residue_index, atom_to_residue])
                link_type.extend([residue_to_atom, atom_to_residue])

            previous_atom_ids = atom_ids
            cumsum.append(offset)

        if contact_edges:
            contact_relation = cls.relation2id["residue_contact"]
            for src, dst in contact_edges:
                if src == dst or src >= num_residue or dst >= num_residue:
                    continue
                edge_list.extend([[src, dst, contact_relation], [dst, src, contact_relation]])
                link_type.extend([contact_relation, contact_relation])

        unit_type_tensor = torch.tensor(unit_type, dtype=torch.long)
        link_type_tensor = torch.tensor(link_type, dtype=torch.long)
        unit_feature = torch.stack(node_features, dim=0).float()
        link_feature = F.one_hot(link_type_tensor, num_classes=len(cls.relation_types)).float()

        return cls(
            edge_list=edge_list,
            unit_type=unit_type_tensor,
            link_type=link_type_tensor,
            unit_feature=unit_feature,
            link_feature=link_feature,
            num_node=len(unit_type),
            num_relation=len(cls.relation_types),
            cumsum=cumsum,
            sequence=sequence,
        )

    @classmethod
    def _build_pseudo_graph(cls, sequence, contact_edges=None):
        num_residue = len(sequence)
        unit_type = [cls.residue2id[token] for token in sequence]
        node_features = [cls._residue_feature(token, idx, num_residue) for idx, token in enumerate(sequence)]
        edge_list = []
        link_type = []

        for residue_index in range(num_residue - 1):
            src = residue_index
            dst = residue_index + 1
            relation = cls.relation2id["residue_backbone"]
            edge_list.extend([[src, dst, relation], [dst, src, relation]])
            link_type.extend([relation, relation])

        cumsum = [num_residue]
        offset = num_residue

        for residue_index, residue in enumerate(sequence):
            atom_names = ["N", "CA", "C", "O"]
            if residue != "G":
                atom_names.append("CB")
            element_lookup = {atom_name: BACKBONE_ELEMENTS.get(atom_name, "C") for atom_name in atom_names}
            atom_descriptors = cls._atom_descriptors(
                residue=residue,
                residue_index=residue_index,
                num_residue=num_residue,
                atom_names=atom_names,
                element_lookup=element_lookup,
                bonds=[
                    ("N", "CA", "single"),
                    ("CA", "C", "single"),
                    ("C", "O", "double"),
                ] + ([("CA", "CB", "single")] if "CB" in atom_names else []),
            )

            local_ids = {}
            for atom_name in atom_names:
                element = element_lookup[atom_name]
                local_ids[atom_name] = offset
                unit_type.append(cls._atom_unit_type(element))
                node_features.append(
                    cls._atom_feature(
                        element,
                        atom_name,
                        residue,
                        residue_index,
                        num_residue,
                        atom_descriptors[atom_name],
                    )
                )
                offset += 1

            for atom_src, atom_dst, bond_type in [
                ("N", "CA", "single"),
                ("CA", "C", "single"),
                ("C", "O", "double"),
            ]:
                relation = cls._relation_from_bond_type(bond_type)
                src = local_ids[atom_src]
                dst = local_ids[atom_dst]
                edge_list.extend([[src, dst, relation], [dst, src, relation]])
                link_type.extend([relation, relation])

            if "CB" in local_ids:
                relation = cls._relation_from_bond_type("single")
                edge_list.extend([[local_ids["CA"], local_ids["CB"], relation], [local_ids["CB"], local_ids["CA"], relation]])
                link_type.extend([relation, relation])

            residue_to_atom = cls.relation2id["residue_to_atom"]
            atom_to_residue = cls.relation2id["atom_to_residue"]
            for atom_id in local_ids.values():
                edge_list.append([residue_index, atom_id, residue_to_atom])
                edge_list.append([atom_id, residue_index, atom_to_residue])
                link_type.extend([residue_to_atom, atom_to_residue])

            cumsum.append(offset)

        if contact_edges:
            contact_relation = cls.relation2id["residue_contact"]
            for src, dst in contact_edges:
                if src == dst or src >= num_residue or dst >= num_residue:
                    continue
                edge_list.extend([[src, dst, contact_relation], [dst, src, contact_relation]])
                link_type.extend([contact_relation, contact_relation])

        unit_type_tensor = torch.tensor(unit_type, dtype=torch.long)
        link_type_tensor = torch.tensor(link_type, dtype=torch.long)
        unit_feature = torch.stack(node_features, dim=0).float()
        link_feature = F.one_hot(link_type_tensor, num_classes=len(cls.relation_types)).float()

        return cls(
            edge_list=edge_list,
            unit_type=unit_type_tensor,
            link_type=link_type_tensor,
            unit_feature=unit_feature,
            link_feature=link_feature,
            num_node=len(unit_type),
            num_relation=len(cls.relation_types),
            cumsum=cumsum,
            sequence=sequence,
        )

    @classmethod
    def from_sequence(cls, sequence, include_pseudo_atoms=None, contact_edges=None, atom_mode="template"):
        sequence = cls.normalize_sequence(sequence)
        if not sequence:
            raise ValueError("Sequence must not be empty")

        if include_pseudo_atoms is not None:
            atom_mode = "pseudo" if include_pseudo_atoms else "none"

        if atom_mode == "none":
            return cls._build_residue_only_graph(sequence, contact_edges)
        if atom_mode == "pseudo":
            return cls._build_pseudo_graph(sequence, contact_edges)
        if atom_mode == "template":
            return cls._build_template_graph(sequence, contact_edges)
        raise ValueError("Unknown atom mode `%s`" % atom_mode)

    @classmethod
    def _build_residue_only_graph(cls, sequence, contact_edges=None):
        num_residue = len(sequence)
        unit_type = [cls.residue2id[token] for token in sequence]
        node_features = [cls._residue_feature(token, idx, num_residue) for idx, token in enumerate(sequence)]
        edge_list = []
        link_type = []

        for residue_index in range(num_residue - 1):
            src = residue_index
            dst = residue_index + 1
            relation = cls.relation2id["residue_backbone"]
            edge_list.extend([[src, dst, relation], [dst, src, relation]])
            link_type.extend([relation, relation])

        if contact_edges:
            contact_relation = cls.relation2id["residue_contact"]
            for src, dst in contact_edges:
                if src == dst or src >= num_residue or dst >= num_residue:
                    continue
                edge_list.extend([[src, dst, contact_relation], [dst, src, contact_relation]])
                link_type.extend([contact_relation, contact_relation])

        unit_type_tensor = torch.tensor(unit_type, dtype=torch.long)
        link_type_tensor = torch.tensor(link_type, dtype=torch.long)
        unit_feature = torch.stack(node_features, dim=0).float()
        link_feature = F.one_hot(link_type_tensor, num_classes=len(cls.relation_types)).float()

        return cls(
            edge_list=edge_list,
            unit_type=unit_type_tensor,
            link_type=link_type_tensor,
            unit_feature=unit_feature,
            link_feature=link_feature,
            num_node=len(unit_type),
            num_relation=len(cls.relation_types),
            cumsum=[num_residue],
            sequence=sequence,
        )

    @classmethod
    def pack(cls, graphs):
        edge_list = []
        edge_weight = []
        num_nodes = []
        num_edges = []
        cumsums = []
        num_relation = -1
        num_cum_node = 0
        num_cum_edge = 0
        num_graph = 0
        data_dict = defaultdict(list)
        meta_dict = graphs[0].meta_dict

        for graph in graphs:
            edge_list.append(graph.edge_list)
            edge_weight.append(graph.edge_weight)
            num_nodes.append(graph.num_node)
            num_edges.append(graph.num_edge)
            cumsums.append(graph.cumsum)

            for key, value in graph.data_dict.items():
                for meta_type in meta_dict[key]:
                    if meta_type == "graph":
                        value = value.unsqueeze(0)
                    elif meta_type == "node reference":
                        value = value + num_cum_node
                    elif meta_type == "edge reference":
                        value = value + num_cum_edge
                    elif meta_type == "graph reference":
                        value = value + num_graph
                data_dict[key].append(value)

            if num_relation == -1:
                num_relation = graph.num_relation
            elif num_relation != graph.num_relation:
                raise ValueError(
                    "Inconsistent `num_relation` in graphs. Expect %d but got %d."
                    % (num_relation, graph.num_relation)
                )

            num_cum_node += graph.num_node
            num_cum_edge += graph.num_edge
            num_graph += 1

        edge_list = torch.cat(edge_list)
        edge_weight = torch.cat(edge_weight)
        data_dict = {key: torch.cat(value) for key, value in data_dict.items()}

        return cls.packed_type(
            edge_list=edge_list,
            edge_weight=edge_weight,
            num_relation=num_relation,
            num_nodes=num_nodes,
            num_edges=num_edges,
            cumsums=cumsums,
            meta_dict=meta_dict,
            **data_dict,
        )

    def unit(self):
        return self.node()

    def link(self):
        return self.edge()

    def peptide(self):
        return self.graph()

    @property
    def node_feature(self):
        return self.unit_feature

    @node_feature.setter
    def node_feature(self, value):
        self.unit_feature = value

    @property
    def edge_feature(self):
        return self.link_feature

    @edge_feature.setter
    def edge_feature(self, value):
        self.link_feature = value

    def append_residue_embedding(self, residue_embedding):
        residue_embedding = torch.as_tensor(residue_embedding, dtype=torch.float, device=self.device)
        if residue_embedding.ndim != 2:
            raise ValueError("`residue_embedding` should be 2D, but found shape %s" % (tuple(residue_embedding.shape),))

        if self.cumsum:
            num_residue = int(self.cumsum[0])
        else:
            residue_cutoff = len(self.residue_types)
            num_residue = int((self.unit_type < residue_cutoff).sum().item())

        if residue_embedding.shape[0] != num_residue:
            raise ValueError(
                "Residue embedding length mismatch. Expect %d but found %d"
                % (num_residue, residue_embedding.shape[0])
            )

        base_feature = self.node_feature.float()
        plm_feature = torch.zeros(self.num_node, residue_embedding.shape[1], dtype=base_feature.dtype, device=self.device)
        plm_feature[:num_residue] = residue_embedding.to(base_feature.dtype)
        self.node_feature = torch.cat([base_feature, plm_feature], dim=-1)
        return self

    def __repr__(self):
        fields = ["num_node=%d" % self.num_node, "num_edge=%d" % self.num_edge]
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))


class PackedPeptideGraph(PackedGraph, HeterogeneousPeptideGraph):
    unpacked_type = HeterogeneousPeptideGraph
    unit2graph = PackedGraph.node2graph
    link2graph = PackedGraph.edge2graph

    def __init__(self, edge_list=None, unit_type=None, link_type=None, num_nodes=None, num_edges=None,
                 cumsums=None, offsets=None, **kwargs):
        if "num_relation" not in kwargs:
            kwargs["num_relation"] = len(self.relation_types)
        super().__init__(
            edge_list=edge_list,
            num_nodes=num_nodes,
            num_edges=num_edges,
            offsets=offsets,
            unit_type=unit_type,
            link_type=link_type,
            **kwargs,
        )
        self.cumsums = cumsums or []


HeterogeneousPeptideGraph.packed_type = PackedPeptideGraph
