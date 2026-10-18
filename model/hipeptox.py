from collections.abc import Sequence
import copy

import torch
from torch import nn

from torchdrug import core, layers
from torchdrug.core import Registry as R

from util.data import HeterogeneousPeptideGraph


def _activation_layer(name):
    name = (name or "relu").strip().lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError("Unsupported activation `%s` for PLM branch" % name)


@R.register("models.HiPepTox")
class HiPepTox(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_unit=None, num_relation=None, edge_input_dim=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum",
                 readout_type="residue", use_node_feature=True, separate_node_proj=False,
                 residue_plm_fusion="sum", plm_dual_branch=False, graph_branch_ignore_plm=False,
                 graph_branch_ignore_tail_dim=0, plm_branch_fusion="concat", plm_branch_hidden_dim=None, plm_cross_attention=False,
                 plm_cross_attention_heads=4, plm_cross_attention_dropout=0.1):
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        num_unit = num_unit or HeterogeneousPeptideGraph.num_unit_type()
        num_relation = num_relation or len(HeterogeneousPeptideGraph.relation_types)

        if not hidden_dims or any(dim != input_dim for dim in hidden_dims):
            raise ValueError(
                "HiPepTox hierarchical updates require every hidden dimension to equal "
                "input_dim; got input_dim=%s and hidden_dims=%s" % (input_dim, list(hidden_dims))
            )
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.node_output_dim = self.output_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.use_node_feature = use_node_feature
        self.separate_node_proj = separate_node_proj
        self.residue_plm_fusion = residue_plm_fusion
        self.plm_dual_branch = plm_dual_branch
        self.graph_branch_ignore_plm = graph_branch_ignore_plm
        self.graph_branch_ignore_tail_dim = int(graph_branch_ignore_tail_dim or 0)
        self.plm_branch_fusion = plm_branch_fusion
        self.plm_cross_attention = plm_cross_attention
        self.base_node_feature_dim = HeterogeneousPeptideGraph.node_feature_dim()
        self.plm_branch_hidden_dim = plm_branch_hidden_dim or self.node_output_dim

        self.embedding = nn.Embedding(num_unit, input_dim)
        if use_node_feature:
            if separate_node_proj:
                self.residue_hand_feature_proj = nn.Linear(self.base_node_feature_dim, input_dim)
                self.atom_feature_proj = nn.Linear(self.base_node_feature_dim, input_dim)
                self.residue_plm_proj = nn.LazyLinear(input_dim)
                self.node_feature_proj = None
                if residue_plm_fusion == "gate":
                    self.residue_plm_gate = nn.Sequential(
                        nn.Linear(input_dim * 2, input_dim),
                        nn.Sigmoid(),
                    )
                elif residue_plm_fusion != "sum":
                    raise ValueError("Unknown residue PLM fusion `%s`" % residue_plm_fusion)
            else:
                self.node_feature_proj = nn.LazyLinear(input_dim)
                self.residue_hand_feature_proj = None
                self.atom_feature_proj = None
                self.residue_plm_proj = None
        else:
            self.node_feature_proj = None
            self.residue_hand_feature_proj = None
            self.atom_feature_proj = None
            self.residue_plm_proj = None
        self.atom_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.residue_layers = nn.ModuleList()

        for i in range(len(self.dims) - 1):
            self.atom_layers.append(
                layers.RelationalGraphConv(
                    self.dims[i], self.dims[i + 1], num_relation, edge_input_dim, batch_norm, activation
                )
            )
            self.cross_layers.append(
                layers.RelationalGraphConv(
                    self.dims[i], self.dims[i + 1], num_relation, edge_input_dim, batch_norm, activation
                )
            )
            self.residue_layers.append(
                layers.RelationalGraphConv(
                    self.dims[i], self.dims[i + 1], num_relation, edge_input_dim, batch_norm, activation
                )
            )

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "dual":
            self.readout = layers.MeanReadout()
            self.readout_ext = layers.MaxReadout()
            self.output_dim = self.output_dim * 2
        else:
            raise ValueError("Unknown readout `%s`" % readout)

        self.graph_output_dim = self.output_dim

        self.readout_type = readout_type
        if readout_type not in {"residue", "all"}:
            raise ValueError("Unknown readout type `%s`" % readout_type)

        if self.plm_dual_branch:
            if plm_branch_fusion not in {"concat", "sum", "gate"}:
                raise ValueError("Unknown PLM branch fusion `%s`" % plm_branch_fusion)

            self.plm_branch_proj = nn.Sequential(
                nn.LazyLinear(self.plm_branch_hidden_dim),
                _activation_layer(activation),
                nn.Linear(self.plm_branch_hidden_dim, self.plm_branch_hidden_dim),
                _activation_layer(activation),
            )
            if readout == "sum":
                self.plm_readout = layers.SumReadout()
            else:
                self.plm_readout = layers.MeanReadout()
            if readout == "dual":
                self.plm_readout_ext = layers.MaxReadout()
                self.plm_output_dim = self.plm_branch_hidden_dim * 2
            else:
                self.plm_readout_ext = None
                self.plm_output_dim = self.plm_branch_hidden_dim

            if self.plm_cross_attention:
                self.plm_cross_attention_proj = nn.Identity()
                if self.plm_branch_hidden_dim != self.node_output_dim:
                    self.plm_cross_attention_proj = nn.Linear(self.plm_branch_hidden_dim, self.node_output_dim)
                self.plm_cross_attention_layer = nn.MultiheadAttention(
                    self.node_output_dim,
                    plm_cross_attention_heads,
                    dropout=plm_cross_attention_dropout,
                    batch_first=True,
                )
                self.plm_cross_attention_norm1 = nn.LayerNorm(self.node_output_dim)
                self.plm_cross_attention_ffn = nn.Sequential(
                    nn.Linear(self.node_output_dim, self.node_output_dim * 4),
                    _activation_layer(activation),
                    nn.Linear(self.node_output_dim * 4, self.node_output_dim),
                )
                self.plm_cross_attention_norm2 = nn.LayerNorm(self.node_output_dim)
                self.plm_fusion_proj = None
                self.plm_fusion_gate = None
                self.output_dim = self.graph_output_dim
            elif plm_branch_fusion == "concat":
                self.plm_cross_attention_proj = None
                self.plm_cross_attention_layer = None
                self.plm_cross_attention_norm1 = None
                self.plm_cross_attention_ffn = None
                self.plm_cross_attention_norm2 = None
                self.plm_fusion_proj = None
                self.plm_fusion_gate = None
                self.output_dim = self.graph_output_dim + self.plm_output_dim
            else:
                self.plm_cross_attention_proj = None
                self.plm_cross_attention_layer = None
                self.plm_cross_attention_norm1 = None
                self.plm_cross_attention_ffn = None
                self.plm_cross_attention_norm2 = None
                self.plm_fusion_proj = nn.Linear(self.plm_output_dim, self.graph_output_dim)
                if plm_branch_fusion == "gate":
                    self.plm_fusion_gate = nn.Sequential(
                        nn.Linear(self.graph_output_dim * 2, self.graph_output_dim),
                        nn.Sigmoid(),
                    )
                else:
                    self.plm_fusion_gate = None
                self.output_dim = self.graph_output_dim
        else:
            self.plm_branch_proj = None
            self.plm_readout = None
            self.plm_readout_ext = None
            self.plm_output_dim = 0
            self.plm_fusion_proj = None
            self.plm_fusion_gate = None
            self.plm_cross_attention_proj = None
            self.plm_cross_attention_layer = None
            self.plm_cross_attention_norm1 = None
            self.plm_cross_attention_ffn = None
            self.plm_cross_attention_norm2 = None

    def _project_node_feature(self, graph, is_residue_node, is_atom_node):
        node_feature = graph.node_feature.float()
        if not self.separate_node_proj:
            if self.graph_branch_ignore_plm and node_feature.shape[-1] > self.base_node_feature_dim:
                if self.graph_branch_ignore_tail_dim > 0 and node_feature.shape[-1] > self.graph_branch_ignore_tail_dim:
                    node_feature = node_feature[:, :-self.graph_branch_ignore_tail_dim]
                else:
                    node_feature = node_feature[:, :self.base_node_feature_dim]
            return self.node_feature_proj(node_feature)

        hand_feature = node_feature[:, :self.base_node_feature_dim]
        projected = torch.zeros(
            node_feature.shape[0], self.input_dim, dtype=node_feature.dtype, device=node_feature.device
        )

        if is_atom_node.any():
            projected[is_atom_node] = self.atom_feature_proj(hand_feature[is_atom_node])

        if is_residue_node.any():
            residue_feature = self.residue_hand_feature_proj(hand_feature[is_residue_node])
            if node_feature.shape[-1] > self.base_node_feature_dim:
                plm_feature = node_feature[is_residue_node, self.base_node_feature_dim:]
                projected_plm = self.residue_plm_proj(plm_feature)
                if self.residue_plm_fusion == "sum":
                    residue_feature = residue_feature + projected_plm
                else:
                    gate = self.residue_plm_gate(torch.cat([residue_feature, projected_plm], dim=-1))
                    residue_feature = residue_feature + gate * projected_plm
            projected[is_residue_node] = residue_feature

        return projected

    def _pool_residue_feature(self, graph, is_residue_node, residue_feature):
        subgraph = graph.node_mask(is_residue_node, compact=True)
        graph_feature = self.readout(subgraph, residue_feature)
        if hasattr(self, "readout_ext"):
            graph_feature = torch.cat([graph_feature, self.readout_ext(subgraph, residue_feature)], dim=-1)
        return graph_feature

    def _pool_plm_branch(self, graph, is_residue_node):
        residue_plm_feature = self._project_plm_residue_feature(graph, is_residue_node)
        subgraph = graph.node_mask(is_residue_node, compact=True)
        plm_graph_feature = self.plm_readout(subgraph, residue_plm_feature)
        if self.plm_readout_ext is not None:
            plm_graph_feature = torch.cat([plm_graph_feature, self.plm_readout_ext(subgraph, residue_plm_feature)], dim=-1)
        return plm_graph_feature

    def _project_plm_residue_feature(self, graph, is_residue_node):
        if graph.node_feature.shape[-1] <= self.base_node_feature_dim:
            raise ValueError("PLM branch requires appended PLM residue embeddings in `graph.node_feature`")

        residue_plm_feature = graph.node_feature[is_residue_node, self.base_node_feature_dim:].float()
        return self.plm_branch_proj(residue_plm_feature)

    def _residue_counts(self, graph, is_residue_node):
        if hasattr(graph, "node2graph") and hasattr(graph, "num_nodes"):
            num_graph = int(graph.num_nodes.shape[0])
            node2graph = graph.node2graph[is_residue_node]
            return torch.bincount(node2graph, minlength=num_graph).tolist()
        return [int(is_residue_node.sum().item())]

    def _pad_residue_feature(self, residue_feature, counts):
        batch_size = len(counts)
        max_length = max(counts) if counts else 0
        padded = residue_feature.new_zeros((batch_size, max_length, residue_feature.shape[-1]))
        mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=residue_feature.device)

        start = 0
        for graph_index, count in enumerate(counts):
            if count:
                padded[graph_index, :count] = residue_feature[start:start + count]
                mask[graph_index, :count] = True
                start += count
        return padded, mask

    def _flatten_residue_feature(self, padded_feature, counts):
        pieces = []
        for graph_index, count in enumerate(counts):
            if count:
                pieces.append(padded_feature[graph_index, :count])
        if pieces:
            return torch.cat(pieces, dim=0)
        return padded_feature.new_zeros((0, padded_feature.shape[-1]))

    def _cross_attend_residue_feature(self, graph, is_residue_node, residue_feature, residue_plm_feature):
        counts = self._residue_counts(graph, is_residue_node)
        query, mask = self._pad_residue_feature(residue_feature, counts)
        key_value, _ = self._pad_residue_feature(residue_plm_feature, counts)
        key_value = self.plm_cross_attention_proj(key_value)

        attention_output, _ = self.plm_cross_attention_layer(
            query,
            key_value,
            key_value,
            key_padding_mask=~mask,
            need_weights=False,
        )
        fused = self.plm_cross_attention_norm1(query + attention_output)
        fused = self.plm_cross_attention_norm2(fused + self.plm_cross_attention_ffn(fused))
        return self._flatten_residue_feature(fused, counts)

    def forward(self, graph, input, all_loss=None, metric=None):
        del input, all_loss, metric

        residue_cutoff = len(HeterogeneousPeptideGraph.residue_types)
        is_atom_node = graph.unit_type >= residue_cutoff
        is_residue_node = ~is_atom_node

        node_in, node_out, _ = graph.edge_list.t()
        node_in_type = graph.unit_type[node_in]
        node_out_type = graph.unit_type[node_out]

        atom_graph = copy.deepcopy(graph)
        atom_mask = (node_in_type >= residue_cutoff) & (node_out_type >= residue_cutoff)
        atom_graph._edge_list = graph.edge_list[atom_mask]
        atom_graph._edge_weight = graph.edge_weight[atom_mask]
        atom_graph.edge_feature = graph.edge_feature[atom_mask]

        cross_graph = copy.deepcopy(graph)
        cross_mask = (node_in_type >= residue_cutoff) & (node_out_type < residue_cutoff)
        cross_graph._edge_list = graph.edge_list[cross_mask]
        cross_graph._edge_weight = graph.edge_weight[cross_mask]
        cross_graph.edge_feature = graph.edge_feature[cross_mask]

        residue_graph = copy.deepcopy(graph)
        residue_mask = (node_in_type < residue_cutoff) & (node_out_type < residue_cutoff)
        residue_graph._edge_list = graph.edge_list[residue_mask]
        residue_graph._edge_weight = graph.edge_weight[residue_mask]
        residue_graph.edge_feature = graph.edge_feature[residue_mask]

        layer_input = self.embedding(graph.unit_type.clamp(min=0))
        if self.use_node_feature and hasattr(graph, "node_feature") and graph.node_feature is not None:
            layer_input = layer_input + self._project_node_feature(graph, is_residue_node, is_atom_node)
        hiddens = []

        for idx in range(len(self.atom_layers)):
            hidden_atom = self.atom_layers[idx](atom_graph, layer_input)

            mixed_input = torch.zeros_like(layer_input)
            mixed_input[is_atom_node] = hidden_atom[is_atom_node]
            mixed_input[is_residue_node] = layer_input[is_residue_node]
            hidden_cross = self.cross_layers[idx](cross_graph, mixed_input)

            residue_input = torch.zeros_like(layer_input)
            residue_input[is_residue_node] = hidden_cross[is_residue_node]
            hidden_residue = self.residue_layers[idx](residue_graph, residue_input)

            layer_output = torch.zeros_like(layer_input)
            layer_output[is_atom_node] = hidden_atom[is_atom_node]
            layer_output[is_residue_node] = hidden_residue[is_residue_node]
            if self.short_cut:
                layer_output = layer_output + layer_input

            hiddens.append(layer_output)
            layer_input = layer_output

        if self.concat_hidden:
            all_node_feature = torch.cat(hiddens, dim=-1)
        else:
            all_node_feature = hiddens[-1]

        residue_node_feature = all_node_feature[is_residue_node]
        if self.readout_type == "residue":
            graph_feature = self._pool_residue_feature(graph, is_residue_node, residue_node_feature)
        else:
            graph_feature = self.readout(graph, all_node_feature)
            if hasattr(self, "readout_ext"):
                graph_feature = torch.cat([graph_feature, self.readout_ext(graph, all_node_feature)], dim=-1)

        graph_branch_feature = graph_feature
        plm_graph_feature = None
        if self.plm_dual_branch:
            residue_plm_feature = self._project_plm_residue_feature(graph, is_residue_node)
            subgraph = graph.node_mask(is_residue_node, compact=True)
            plm_graph_feature = self.plm_readout(subgraph, residue_plm_feature)
            if self.plm_readout_ext is not None:
                plm_graph_feature = torch.cat([plm_graph_feature, self.plm_readout_ext(subgraph, residue_plm_feature)], dim=-1)

            if self.plm_cross_attention:
                residue_node_feature = self._cross_attend_residue_feature(
                    graph, is_residue_node, residue_node_feature, residue_plm_feature
                )
                if self.readout_type == "residue":
                    graph_feature = self._pool_residue_feature(graph, is_residue_node, residue_node_feature)
                else:
                    fused_all_node_feature = all_node_feature.clone()
                    fused_all_node_feature[is_residue_node] = residue_node_feature
                    graph_feature = self.readout(graph, fused_all_node_feature)
                    if hasattr(self, "readout_ext"):
                        graph_feature = torch.cat(
                            [graph_feature, self.readout_ext(graph, fused_all_node_feature)], dim=-1
                        )
            elif self.plm_branch_fusion == "concat":
                graph_feature = torch.cat([graph_feature, plm_graph_feature], dim=-1)
            else:
                projected_plm = self.plm_fusion_proj(plm_graph_feature)
                if self.plm_branch_fusion == "sum":
                    graph_feature = graph_feature + projected_plm
                else:
                    gate = self.plm_fusion_gate(torch.cat([graph_feature, projected_plm], dim=-1))
                    graph_feature = graph_feature + gate * projected_plm

        return {
            "graph_feature": graph_feature,
            "graph_branch_feature": graph_branch_feature,
            "plm_graph_feature": plm_graph_feature,
            "node_feature": residue_node_feature,
            "all_node_feature": all_node_feature,
        }
