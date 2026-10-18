import os
import torch
import zipfile
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from torch_geometric.data import Data, Dataset
from peptideGraph import ProteinGraphBuilder
from utils import read_fasta, merge
import torch.nn.functional as F
from embeddings import ESM2_Embeddings
from ESMfolding import esm2fold
from typing import Dict
import warnings

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def _build_graph_worker(record: Dict, pdbs_path: str, max_d: int) -> Dict:
    graph_builder = ProteinGraphBuilder(max_distance=max_d, max_num_neighbors=128)
    pdbs_zip = os.path.join(pdbs_path, f"{record['id']}.result.zip")
    pdb_file = os.path.join(pdbs_path, f"{record['id']}.pdb")

    try:
        if os.path.isfile(pdbs_zip):
            with zipfile.ZipFile(pdbs_zip, "r") as zip_ref:
                for file in zip_ref.namelist():
                    if "_relaxed_" in file and file.endswith('.pdb'):
                        extracted_pdb = zip_ref.extract(file, path=pdbs_path)
                        graph = graph_builder.build_graph(
                            id=record['id'],
                            pdb_file=extracted_pdb,
                            amd=record['AMD'],
                            label=record['label'],
                        )
                        return {"graph": graph, "error": None, "id": record["id"]}
            return {"graph": None, "error": f"No relaxed PDB found in {pdbs_zip}", "id": record["id"]}

        if os.path.isfile(pdb_file):
            graph = graph_builder.build_graph(
                id=record['id'],
                pdb_file=pdb_file,
                amd=record['AMD'],
                label=record['label'],
            )
            return {"graph": graph, "error": None, "id": record["id"]}

        return {
            "graph": None,
            "error": f"Missing structure for {record['id']}: expected {pdbs_zip} or {pdb_file}",
            "id": record["id"],
        }
    except Exception as exc:
        return {"graph": None, "error": f"{record['id']}: {exc}", "id": record["id"]}


class ToxicityData(Dataset):
    def __init__(self,
                 seqs: str = None,
                 pos_seqs: str = None,
                 neg_seqs: str = None,
                 pdbs_path: str = None,
                 max_d: int = 12,
                 device: torch.device = torch.device('cpu'),
                 prediction_tool: str = 'ColabFold',
                 embedding_model: str = 't12',) -> None:

        super().__init__()
        self.graphs = list()
        self.pdbs_path = pdbs_path
        self.max_d = max_d
        self.embedding_model = embedding_model
        self.graph_builder = ProteinGraphBuilder(max_distance=max_d,
                                                 max_num_neighbors=128)
        self.esm = None
        self.structure_workers = max(
            1,
            int(os.environ.get("TAMPER_STRUCTURE_WORKERS", min(8, os.cpu_count() or 1))),
        )

        if pos_seqs and neg_seqs:
            self.data = merge(pos_fasta=pos_seqs, neg_fasta=neg_seqs)
        else:
            self.data = read_fasta(fasta_file=seqs)

        logger.info(f"Loading structures from {pdbs_path}")

        self.add_structures(tool=prediction_tool,
                            device=device)
        
        logger.info(f"Number of structures: {len(self.graphs)}")

        logger.info(f"Generating sequence embeddings")
        self.esm = ESM2_Embeddings(model_variant=self.embedding_model)
        self.load_embeddings()

    def load_embeddings(self,):

        seqs, ids = [], []

        for record in self.data:
            seqs.append(record['seq'])
            ids.append(record['id'])

        embeddings = self.esm.generate_embeddings(sequences=seqs, ids=ids)

        for graph in self.graphs:
            graph.embeddings = embeddings[graph.id]

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]

    def add_structures(self,
                       tool: str = 'ColabFold',
                       device: torch.device = torch.device('cpu')) -> None:

        if tool == 'ColabFold':
            total = len(self.data)
            logger.info(f"Building graphs with {self.structure_workers} worker(s)")

            if self.structure_workers == 1:
                iterator = map(_build_graph_worker, self.data, repeat(self.pdbs_path), repeat(self.max_d))
            else:
                executor = ProcessPoolExecutor(max_workers=self.structure_workers)
                iterator = executor.map(_build_graph_worker, self.data, repeat(self.pdbs_path), repeat(self.max_d))

            skipped = 0
            try:
                for index, result in enumerate(iterator, start=1):
                    if result["graph"] is not None:
                        self.graphs.append(result["graph"])
                    else:
                        skipped += 1
                        logger.warning(f"Skipping structure {result['id']}: {result['error']}")

                    if index % 250 == 0 or index == total:
                        logger.info(
                            f"Processed {index}/{total} structures; "
                            f"kept={len(self.graphs)} skipped={skipped}"
                        )
            finally:
                if self.structure_workers != 1:
                    executor.shutdown(wait=True, cancel_futures=True)
        else:
            esm2fold(my_data=self.data, result_file=self.pdbs_path, device=device)
            
            for index in range(len(self.data)):
                pdb_file = os.path.join(self.pdbs_path, f"{self.data[index]['id']}.pdb")
                self.graphs.append(self.graph_builder.build_graph(
                                   id=self.data[index]['id'],
                                   pdb_file=pdb_file,
                                   amd=self.data[index]['AMD'],
                                   label=self.data[index]['label']))
