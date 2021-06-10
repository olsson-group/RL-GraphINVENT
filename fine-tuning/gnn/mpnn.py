# load general packages and functions
import math
import torch

# load program-specific functions
import gnn.summation_mpnn
import gnn.modules

# defines specific MPNN implementations


# some constants
BIG_NEGATIVE = -1e6
BIG_POSITIVE = 1e6


class GGNN(gnn.summation_mpnn.SummationMPNN):
    """ The "gated-graph neural network" model.

    Args:
      *edge_features (int) : Number of edge features.
      enn_depth (int) : Num layers in 'enn' MLP.
      enn_dropout_p (float) : Dropout probability in 'enn' MLP.
      enn_hidden_dim (int) : Number of weights (layer width) in 'enn' MLP.
      *f_add_elems (int) : Number of elements PER NODE in `f_add` (e.g.
        `n_atom_types` * `n_formal_charge` * `n_edge_features`).
      mlp1_depth (int) : Num layers in first-tier MLP in APD readout function.
      mlp1_dropout_p (float) : Dropout probability in first-tier MLP in APD
        readout function.
      mlp1_hidden_dim (int) : Number of weights (layer width) in first-tier MLP
        in APD readout function.
      mlp2_depth (int) : Num layers in second-tier MLP in APD readout function.
      mlp2_dropout_p (float) : Dropout probability in second-tier MLP in APD
        readout function.
      mlp2_hidden_dim (int) : Number of weights (layer width) in second-tier MLP
        in APD readout function.
      gather_att_depth (int) : Num layers in 'gather_att' MLP in graph gather block.
      gather_att_dropout_p (float) : Dropout probability in 'gather_att' MLP in
        graph gather block.
      gather_att_hidden_dim (int) : Number of weights (layer width) in
        'gather_att' MLP in graph gather block.
      gather_emb_depth (int) : Num layers in 'gather_emb' MLP in graph gather block.
      gather_emb_dropout_p (float) : Dropout probability in 'gather_emb' MLP in
        graph gather block.
      gather_emb_hidden_dim (int) : Number of weights (layer width) in
        'gather_emb' MLP in graph gather block.
      gather_width (int) : Output size of graph gather block block.
      hidden_node_features (int) : Indicates length of node hidden states.
      *initialization (str) : Initialization scheme for weights in feed-forward
        networks ('none', 'uniform', or 'normal').
      message_passes (int) : Number of message passing steps.
      message_size (int) : Size of message passed (output size of all MLPs in
        message aggregation step, input size to `GRU`).
      *n_nodes_largest_graph (int) : Number of nodes in the largest graph.
      *node_features (int) : Number of node features (e.g. `n_atom_types` +
        `n_formal_charge`).
    """
    def __init__(self, edge_features, enn_depth, enn_dropout_p, enn_hidden_dim,
                 f_add_elems, mlp1_depth, mlp1_dropout_p, mlp1_hidden_dim,
                 mlp2_depth, mlp2_dropout_p, mlp2_hidden_dim, gather_att_depth,
                 gather_att_dropout_p, gather_att_hidden_dim, gather_width,
                 gather_emb_depth, gather_emb_dropout_p, gather_emb_hidden_dim,
                 hidden_node_features, initialization, message_passes,
                 message_size, n_nodes_largest_graph, node_features):

        super(GGNN, self).__init__(node_features, hidden_node_features, edge_features, message_size, message_passes)

        self.n_nodes_largest_graph = n_nodes_largest_graph

        self.msg_nns = torch.nn.ModuleList()
        for _ in range(edge_features):
            self.msg_nns.append(
                gnn.modules.MLP(
                    in_features=hidden_node_features,
                    hidden_layer_sizes=[enn_hidden_dim] * enn_depth,
                    out_features=message_size,
                    init=initialization,
                    dropout_p=enn_dropout_p,
                )
            )

        self.gru = torch.nn.GRUCell(
            input_size=message_size, hidden_size=hidden_node_features, bias=True
        )

        self.gather = gnn.modules.GraphGather(
            node_features=node_features,
            hidden_node_features=hidden_node_features,
            out_features=gather_width,
            att_depth=gather_att_depth,
            att_hidden_dim=gather_att_hidden_dim,
            att_dropout_p=gather_att_dropout_p,
            emb_depth=gather_emb_depth,
            emb_hidden_dim=gather_emb_hidden_dim,
            emb_dropout_p=gather_emb_dropout_p,
            init=initialization,
        )

        self.APDReadout = gnn.modules.GlobalReadout(
            node_emb_size=hidden_node_features,
            graph_emb_size=gather_width,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            init=initialization,
            f_add_elems=f_add_elems,
            f_conn_elems=edge_features,
            f_term_elems=1,
            max_n_nodes=n_nodes_largest_graph,
        )

    def message_terms(self, nodes, node_neighbours, edges):
        edges_v = edges.view(-1, self.edge_features, 1)
        node_neighbours_v = edges_v * node_neighbours.view(-1, 1, self.hidden_node_features)
        terms_masked_per_edge = [
            edges_v[:, i, :] * self.msg_nns[i](node_neighbours_v[:, i, :])
            for i in range(self.edge_features)
        ]
        return sum(terms_masked_per_edge)

    def update(self, nodes, messages):
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        output = self.APDReadout(hidden_nodes, graph_embeddings)

        return output