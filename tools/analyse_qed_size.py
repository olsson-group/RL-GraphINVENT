from rdkit.Chem import MolFromSmiles, QED
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from utils import load_molecules


font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 15};

mpl.rc('font', **font)

"""
Generates a plot of the QED values of the molecules in a file against their number of nodes

To use script, run:
python tools/analyse_qed_size.py --smi path/to/file.smi
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


def qed_n_atoms(smi_file):
    """ Determines the maximum number of atoms per molecule in an input SMILES file.

    Args:
      smi_file (str) : Full path/filename to SMILES file.
    """
    molecules = load_molecules(path=smi_file)
    qed = []
    n_atoms = []
    idx = 0
    for mol in molecules:
        qed.append(QED.qed(mol))
        n_atoms.append(mol.GetNumAtoms())
        idx += 1
        if idx > 1000:
            break


    qed = np.array(qed)
    n_atoms = np.array(n_atoms, dtype=int)
    qed_avg = []
    
    for n in set(n_atoms):
        mask = np.where(n_atoms == n, np.ones(len(n_atoms), dtype=bool), np.zeros(len(n_atoms), dtype=bool))
        avg = np.sum(qed[mask]) / np.sum(mask)
        qed_avg.append(avg)

    qed_avg = np.array(qed_avg)
    set_n_atoms = np.array([n for n in set(n_atoms)], dtype=int)

    plt.figure()
    plt.scatter(n_atoms, qed, alpha=0.5, label='Data', s=10, edgecolors='none')
    plt.plot(set_n_atoms, qed_avg, c='r', label='Avg.')
    plt.xlabel("Number of atoms")
    plt.ylabel("QED")
    plt.legend()
    plt.show()
    plt.savefig("qed-vs-nodes.png")



    


if __name__ == "__main__":
    qed_n_atoms(smi_file=args.smi)
    
    print("Done.", flush=True)