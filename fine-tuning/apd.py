# load general packages and functions
import torch
import copy

# load program-specific functions
from parameters.constants import constants as C

# defines functions related to the action probability distributions (APDs)



def get_decoding_route_length(molecular_graph):
    """ Returns the number of subgraphs in the `molecular_graph`s decoding route,
    which is how many subgraphs would be formed in the process of deleting the
    last atom/bond in the molecule stepwise until only a single atom is left.
    Note that this is simply the number of edges plus two.

    Args:
      molecular_graph (PreprocessingGraph) : Molecule to be decoded.

    Returns:
      n_decoding_graphs (int) : Number of subgraphs in the `molecular_graph`s
        decoding route.
    """
    return molecular_graph.get_n_edges() + 2



def split_APD_vector(APD_output, as_vec):
    """ Reshapes and splits the flat APD tensor into separate `f_add`, `f_conn`,
    and `f_term` tensors.

    Args:
      APD_output (torch.Tensor) : Policy tensor with dimensions
        |N|x|VxAxFxHxCxE + VxE + 1|. Rows in the APD tensor correspond to subgraphs
        in the batch, and columns are elements of the flattened APD matrices.
      as_vec (bool) : Indicates whether to shape the split APD tensors as
        vectors (without reshaping) or not.

    Returns:
      f_add (torch.Tensor) : If `as_vec`==False, a tensor of size
        |N|x|V|x|A|x|F|x|H|x|C|x|E|. Otherwise, a tensor of size |N|x|VxAxFxHxCxE|.
        Contains the probabilities of adding a new node to an existing node
        in each subgraph in the batch.
      f_conn (torch.Tensor) : If `as_vec`==False, a tensor of size |N|x|V|x|E|,
        otherwise a tensor of size |n|x|VxE|. Contains the probabilities of
        bonding the most recently added node with another existing node in each
        subgraph in the batch.
      f_term (torch.Tensor) : Tensor of size |N|x|1|. Contains the probabilities
        of terminating each subgraph in the batch.

      V is the set of nodes in the largest graph, P is the set of APDs (the
      number of subgraphs), A is the set of atom types, F is the set of formal
      charges, H is the set of implicit Hs, C is the set of chiral states,
      and E is the set of edge types. N is the batch size.
    """
    # get the number of elements which should be in each APD component
    f_add_elems = int(C.dim_f_add_p0)
    f_conn_elems = int(C.dim_f_conn_p0)

    # split up the target vector into three and reshape
    f_add, f_conn_and_term = torch.split(APD_output, f_add_elems, dim=1)
    f_conn, f_term = torch.split(f_conn_and_term, f_conn_elems, dim=1)

    if as_vec:  # output as vectors, without reshaping
        return f_add, f_conn, f_term

    else:  # reshape the APDs from vectors into proper tensors
        f_add = f_add.view(C.dim_f_add)
        f_conn = f_conn.view(C.dim_f_conn)
        f_term = f_term.view(C.dim_f_term)

        return f_add, f_conn, f_term
