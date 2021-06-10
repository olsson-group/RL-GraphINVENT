import argparse
import rdkit
from utils import load_molecules
import matplotlib.pyplot as plt
import numpy as np

"""
Creates a new dataset which does not have entries corresponding to groups of molecules.

To use script, run:
python tools/remove_groups.py --smi path/to/file.smi
"""

# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--smi",
                    type=str,
                    default="data/pre-training/chembl25_500k/train.smi",
                    help="SMILES file containing molecules to analyse.")
args = parser.parse_args()


def get_groups_removed(smi_file):
    """ Determines the maximum number of atoms per molecule in an input SMILES file.

    Args:
      smi_file (str) : Full path/filename to SMILES file.
    """

    n_mols = 0

    with open(smi_file[:-4] + '_final-selection.smi', 'w') as f_out, open(smi_file, 'r') as f_in:

        for i, line in enumerate(f_in):
            if "." in line:
                continue
            else:
                f_out.write(line)
                n_mols += 1

    return n_mols


if __name__ == "__main__":
    n_mols = get_groups_removed(smi_file=args.smi)
    print("* Number of molecules in output file:", n_mols, flush=True)
    print("Done.", flush=True)
