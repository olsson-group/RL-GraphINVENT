import argparse
import csv

"""
Computes the intersection between two .smi files.

To use script, run:
python tools/compute_intersection.py --smi path/to/file.smi
"""

# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--smi1",
                    type=str,
                    default="data/pre-training/chembl25_500k/train.smi",
                    help="SMILES file 1 containing molecules to analyse.")
parser.add_argument("--smi2",
                    type=str,
                    default="data/fine-tuning/active-mols.smi",
                    help="SMILES file 2 containing molecules to analyse.")
args = parser.parse_args()

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def compute_intersection(smi_file1, smi_file2):


    with open(smi_file1) as f:
        reader = csv.reader(f, delimiter=" ")
        smiles = list(zip(*reader))[0]
    smiles1 = list(smiles)


    with open(smi_file2) as f:
        reader = csv.reader(f, delimiter= " ")
        smiles = list(zip(*reader))[0]
    smiles2 = list(smiles)


    intersection_smiles = intersection(smiles1, smiles2)

    print("SMILES shared:", intersection_smiles, flush=True)
    print(f"SMILES in file 1: {len(smiles1)}", flush=True)
    print(f"SMILES in file 2: {len(smiles2)}", flush=True)
    print(f"Overlap represents {len(intersection_smiles)/len(smiles1)*100:.2f} % of file {smi_file1}.", flush=True)
    print(f"Overlap represents {len(intersection_smiles)/len(smiles2)*100:.2f} % of file {smi_file2}.", flush=True)


if __name__ == "__main__":
    compute_intersection(smi_file1=args.smi1, smi_file2=args.smi2)
    print("Done.", flush=True)






