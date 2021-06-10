import argparse
import rdkit
from utils import load_molecules

"""
Selects a set of molecules with the desired atom types.

To use script, run:
python tools/select_atom_types.py --smi path/to/file.smi --dtb "Name-of-the-database"
"""


# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--smi",
                    type=str,
                    default="data/pre-training/chembl25_500k/train.smi",
                    help="SMILES file containing molecules to analyse.")
parser.add_argument("--dtb",
                    type=str,
                    default="GDB-13",
                    help="Name of the database from which to emulate the set of posible atoms.")
args = parser.parse_args()


def get_atom_types_selected(smi_file, database):
    """ Determines the atom types present in an input SMILES file.

    Args:
      smi_file (str) : Full path/filename to SMILES file.
    """

    # list of atom types to be selected
    if database == "GDB-13":
        atom_types = ['H', 'C', 'N', 'O', 'Cl']
        pt = rdkit.Chem.GetPeriodicTable()
        atom_types = [pt.GetAtomicNumber(atom) for atom in atom_types]
    elif database == "MOSES":
        atom_types = ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br'] #like MOSES
        pt = rdkit.Chem.GetPeriodicTable()
        atom_types = [pt.GetAtomicNumber(atom) for atom in atom_types]
    else:
        raise NotImplementedError

    molecules = load_molecules(path=smi_file)

    n_mols = 0

    filename = smi_file[:-4] + '_selected-atoms.smi'
    smi_writer = rdkit.Chem.rdmolfiles.SmilesWriter(filename, includeHeader=False)
    for mol in molecules:
        flag = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in atom_types:
                flag = 1
                break
        if flag == 0:
            if mol is not None:
                try:
                    smi_writer.write(mol)
                    n_mols += 1
                except:
                    pass

    atom_types = [pt.GetElementSymbol(atom) for atom in atom_types]

    # return the symbols, for convenience
    return n_mols, atom_types


if __name__ == "__main__":
    n_mols, atom_types = get_atom_types_selected(smi_file=args.smi, database=args.dtb)
    print("* Atom types present in output file:", atom_types, flush=True)
    print("* Number of molecues present in the output file:", n_mols, flush=True)
    print("Done.", flush=True)
