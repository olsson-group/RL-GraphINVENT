import argparse
import rdkit
from utils import load_molecules
import matplotlib.pyplot as plt
import numpy as np

"""
Creates a new dataset which only includes molecules smaller than the specified percentile.

To use script, run:
python tools/remove_large_mols.py --smi path/to/file.smi --perc xx
"""

# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--smi",
                    type=str,
                    default="data/pre-training/chembl25_500k/train.smi",
                    help="SMILES file containing molecules to analyse.")
parser.add_argument("--perc",
                    type=float,
                    default=80,
                    help="A float indicating the maximum percentile to consider.")
args = parser.parse_args()


def get_large_mols_removed(smi_file, perc):
    """ Determines the maximum number of atoms per molecule in an input SMILES file.

    Args:
      smi_file (str) : Full path/filename to SMILES file.
    """
    molecules = load_molecules(path=smi_file)
    n_atoms = np.array([mol.GetNumAtoms() for mol in molecules])
    max_n_atoms = np.percentile(n_atoms, perc)
    median_n_atoms = np.median(n_atoms)

    n_mols = 0

    with open(smi_file[:-4] + '_no-large.smi', 'w') as f_out, open(smi_file, 'r') as f_in:

        for i, line in enumerate(f_in):
            if i == 0 and "SMILES" in line:
                n_atoms = np.insert(n_atoms, 0, 0)
                continue
            if n_atoms[i] < max_n_atoms:
                f_out.write(line)
                n_mols += 1

    return int(max_n_atoms), median_n_atoms, n_mols


if __name__ == "__main__":
    max_n_atoms, median_n_atoms, n_mols = get_large_mols_removed(smi_file=args.smi, perc = args.perc)
    print("* Max number of atoms in output file:", max_n_atoms, flush=True)
    print("* Median number of atoms in input file:", median_n_atoms, flush=True)
    print("* Number of molecules in output file:", n_mols, flush=True)
    print("Done.", flush=True)
