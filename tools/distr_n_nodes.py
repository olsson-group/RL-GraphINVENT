import argparse
import rdkit
from utils import load_molecules
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 15};

mpl.rc('font', **font)

"""
Gets the distribution of the number of nodes per molecule present in a set of molecules.

To use script, run:
python tools/distr_n_nodes.py --smi path/to/file.smi
"""

# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--smi",
                    type=str,
                    default="data/pre-training/chembl25_500k/test.smi",
                    help="SMILES file containing molecules to analyse.")
args = parser.parse_args()


def get_distr_n_atoms(smi_file):
    """ Returns the number of atoms per molecule in an input SMILES file.

    Args:
      smi_file (str) : Full path/filename to SMILES file.
    """
    molecules = load_molecules(path=smi_file)

    n_atoms = []
    for mol in molecules:
        n_atoms.append(mol.GetNumAtoms())

    return n_atoms


if __name__ == "__main__":
    n_atoms = get_distr_n_atoms(smi_file=args.smi)
    plt.figure()
    plt.hist(n_atoms, bins = 100)
    plt.xlabel("# atoms")
    plt.ylabel("# molecules")
    plt.savefig("hist.png")
    print("* Max number of atoms in input file:", max(n_atoms), flush=True)
    n_atoms = np.array(n_atoms)
    np.savetxt("n_atoms.txt", n_atoms)
    print("Done.", flush=True)
