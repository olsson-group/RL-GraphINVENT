# load general packages and functions
import csv
import sys

# load program-specific functions
sys.path.insert(1, "./parameters/")  # search "parameters/" directory
import parameters.args as args
import parameters.load as load

# default model parameters, hyperparameters, and settings are defined here
# recommended not to modify the default settings here, but rather create
# an input file with the modified parameters in a new job directory (see README)



# default parameters defined below
""" Defines the default values for job and model parameters. **Alternative to
using argparser, as there are many variables.

General settings for the generative model:
  atom_types (list) : Contains atom types (str) to encode in node features.
  formal_charge (list) : Contains formal charges (int) to encode in node features.
  imp_H (list) : Contains number of implicit hydrogens (int) to encode in node features.
  chirality (list) : Contains chiral states (str) to encode in node features.
  group_size (int) : When preprocessing graphs, this is the size of the
    preprocessing groups (e.g. how many subgraphs preprocessed at once).
  generation_epoch (int) : Epoch to sample during a 'generation' job.
  n_samples (int) : Number of molecules to generate during each sampling epoch.
    Note: if `n_samples` > 100000 molecules, these will be generated in batches
    of 100000.
  n_workers (int) : Number of subprocesses to use during data loading.
  restart (bool) : If specified, will restart training from previous saved state.
    Can only be used for preprocessing or training jobs.
  max_n_nodes (int) : Maximum number of allowed nodes in graph. Must be greater
    than or equal to the number of nodes in largest graph in training set.
  job_type (str) : Options: 'preprocess', 'train', 'generate', or 'test'.
  sample_every (int) : Specifies when to sample the model (i.e. epochs between sampling).
  dataset_dir (str) : Full path to directory containing testing ("test.smi"),
    training ("train.smi"), and validation ("valid.smi") sets.
  use_aromatic_bonds (bool) : If specified, aromatic bond types will be used.
  use_canon (bool) : If specified, uses canonical RDKit ordering in graph representations.
  use_chirality (bool) : If specified, includes chirality in the atomic representations.
  use_explicit_H (bool) : If specified, uses explicit Hs in molecular
    representations (not recommended for most applications).
  ignore_H (bool) : If specified, ignores H's completely in graph representations
    (treats them neither as explicit or implicit). When generating graphs, H's are
    added to graphs after generation is terminated.
  use_tensorboard (bool) : If specified, enables the use of tensorboard during training.
  tensorboard_dir (str) : Path to directory in which to write tensorboard things.
"""
# general job parameters
params_dict = {
    "atom_types": ["C", "N", "O", "S", "Cl"],
    "formal_charge": [-1, 0, 1],
    "imp_H": [0, 1, 2, 3],
    "chirality": ["None", "R", "S"],
    "group_size": 1000,
    "generation_epoch": 30,
    "n_samples": 2000,  #5000,
    "n_workers": 2,
    "restart": False,
    "max_n_nodes": 13,
    "job_type": "train",
    "sample_every": 10,
    "dataset_dir": "../data/fine-tuning/chembl25_500k/",
    "use_aromatic_bonds": False,
    "use_canon": True,
    "use_chirality": False,
    "use_explicit_H": False,
    "ignore_H": True,
    "tensorboard_dir": "tensorboard/",
}
""" MPNN hyperparameters (common ones):
  batch_size (int) : Number of graphs in a mini-batch.
  epochs (int) : Number of training epochs.
  init_lr (float) : Initial learning rate.
  min_rel_lr (float) : Minimum allowed learning rate relative to the initial (used for
    learning rate decay).
  max_rel_lr (float) : Maximum allowed learning rate relative to the initial (used for
    learning rate ramp-up).
  weights_initialization (str) : Initialization scheme for weights in feed-forward networks ('none',
    'uniform', or 'normal').
  model (str) : MPNN model to use ('MNN', 'S2V', 'AttS2V', 'GGNN', 'AttGGNN', or 'EMN').
  weight_decay (float) : Optimizer weight decay (L2 penalty).
"""
# model common hyperparameters
model_common_hp_dict = {
    "batch_size": 64,
    "gen_batch_size": 1000,
    "block_size": 100000,
    "epochs": 100,
    "init_lr": 1e-4,
    "min_rel_lr": 5e-2,
    "max_rel_lr": 1,
    "sigma": 1,
    "weights_initialization": "uniform",
    "weight_decay": 0.0,
    "alpha": 0.5,
}

# make sure job dir ends in "/"
if args.job_dir[-1] != "/":
    print("* Adding '/' to end of `job_dir`.")
    args.job_dir += "/"

# get the model before loading model-specific hyperparameters
try:
    input_csv_path = args.job_dir + "input.csv"
    model = load.which_model(input_csv_path=input_csv_path)
except:
    model = "GGNN"  # default model

model_common_hp_dict["model"] = model


# model-specific hyperparameters (implementation-specific)
if model_common_hp_dict["model"] == "GGNN":
    """ GGNN hyperparameters:
      enn_depth (int) : Num layers in 'enn' MLP.
      enn_dropout_p (float) : Dropout probability in 'enn' MLP.
      enn_hidden_dim (int) : Number of weights (layer width) in 'enn' MLP.
      mlp1_depth (int) : Num layers in first-tier MLP in `APDReadout`.
      mlp1_dropout_p (float) : Dropout probability in first-tier MLP in `APDReadout`.
      mlp1_hidden_dim (int) : Number of weights (layer width) in first-tier MLP in `APDReadout`.
      mlp2_depth (int) : Num layers in second-tier MLP in `APDReadout`.
      mlp2_dropout_p (float) : Dropout probability in second-tier MLP in `APDReadout`.
      mlp2_hidden_dim (int) : Number of weights (layer width) in second-tier MLP in `APDReadout`.
      gather_att_depth (int) : Num layers in 'gather_att' MLP in `GraphGather`.
      gather_att_dropout_p (float) : Dropout probability in 'gather_att' MLP in `GraphGather`.
      gather_att_hidden_dim (int) : Number of weights (layer width) in 'gather_att' MLP in `GraphGather`.
      gather_emb_depth (int) : Num layers in 'gather_emb' MLP in `GraphGather`.
      gather_emb_dropout_p (float) : Dropout probability in 'gather_emb' MLP in `GraphGather`.
      gather_emb_hidden_dim (int) : Number of weights (layer width) in 'gather_emb' MLP in `GraphGather`.
      gather_width (int) : Output size of `GraphGather` block.
      message_passes (int) : Number of message passing steps.
      message_size (int) : Size of message passed (output size of all MLPs in message
        aggregation step, input size to `GRU`).
    """
    model_specific_hp_dict = {
        "enn_depth": 4,
        "enn_dropout_p": 0.0,
        "enn_hidden_dim": 250,
        "mlp1_depth": 4,
        "mlp1_dropout_p": 0.0,
        "mlp1_hidden_dim": 500,
        "mlp2_depth": 4,
        "mlp2_dropout_p": 0.0,
        "mlp2_hidden_dim": 500,
        "gather_att_depth": 4,
        "gather_att_dropout_p": 0.0,
        "gather_att_hidden_dim": 250,
        "gather_emb_depth": 4,
        "gather_emb_dropout_p": 0.0,
        "gather_emb_hidden_dim": 250,
        "gather_width": 100,
        "hidden_node_features": 100,
        "message_passes": 3,
        "message_size": 100,
    }

# make sure dataset dir ends in "/"
if params_dict["dataset_dir"][-1] != "/":
    print("* Adding '/' to end of `dataset_dir`.")
    params_dict["dataset_dir"] += "/"

# join dictionaries
params_dict.update(model_common_hp_dict)
params_dict.update(model_specific_hp_dict)
