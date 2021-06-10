import argparse
import rdkit
from utils import load_molecules

"""
Selects a set of molecules with the desired formal charges.

To use script, run:
python tools/select_formal_charges.py --smi path/to/file.smi
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


def get_formal_charges_selected(smi_file):
    """ Determines the atom types present in an input SMILES file.

    Args:
      smi_file (str) : Full path/filename to SMILES file.
    """

    # list of formal charges to be selected
    formal_charges = [-1, 0, 1]

    molecules = load_molecules(path=smi_file)

    n_mols = 0
    out_file = smi_file[:-4] + "_selected-charges.smi"

    smi_writer = rdkit.Chem.rdmolfiles.SmilesWriter(out_file, includeHeader=False)
    for mol in molecules:
        flag = 0
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() not in formal_charges:
                flag = 1
                break
        if flag == 0:
            if mol is not None:
                try:
                    smi_writer.write(mol)
                    n_mols += 1
                except:
                    pass

    # return the symbols, for convenience
    return n_mols, formal_charges


if __name__ == "__main__":
    n_mols, formal_charges = get_formal_charges_selected(smi_file=args.smi)
    print("* Atom types present in output file:", formal_charges, flush=True)
    print("* Number of molecues present in the output file:", n_mols, flush=True)
    print("Done.", flush=True)
