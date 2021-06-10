import csv
import numpy as np

"""
Recomputes the stats when there are more than 1 generation batch.

To use script, run:
python tools/recompute_stats.py --path path/to/folder/
"""
# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--path",
                    type=str,
                    default='output_chembl25_500k/pre-training/job_0/',
                    help="SMILES file containing molecules to analyse.")

args = parser.parse_args()

def recompute_stats(path):

    with open(path + "regeneration.csv") as f:
        reader = csv.reader(f)
        data = list(zip(*reader))

    n_lines = len(data[0])
    n_models = 32

    fraction_valid = np.zeros((n_models))
    fraction_valid_pt = np.zeros((n_models))
    fraction_pt = np.zeros((n_models))
    avg_n_nodes = np.zeros((n_models))
    avg_n_edges = np.zeros((n_models))
    fraction_unique = np.zeros((n_models))
    epoch_key = []

    epoch = data[0][0]

    idx = 0

    for i_line in range(n_lines):
        if data[0][i_line] == epoch:
            fraction_valid[idx] += float(data[1][i_line])
            fraction_valid_pt[idx] += float(data[2][i_line])*float(data[3][i_line])
            fraction_pt[idx] += float(data[3][i_line])
            avg_n_nodes[idx] += float(data[5][i_line])
            avg_n_edges[idx] += float(data[6][i_line])
        else:
            epoch_key.append(epoch)
            epoch = data[0][i_line]
            idx += 1
            fraction_valid[idx] += float(data[1][i_line])
            fraction_valid_pt[idx] += float(data[2][i_line])*float(data[3][i_line])
            fraction_pt[idx] += float(data[3][i_line])
            avg_n_nodes[idx] += float(data[5][i_line])
            avg_n_edges[idx] += float(data[6][i_line])

    epoch_key.append(epoch)
        
    fraction_valid /= 100
    fraction_valid_pt /= fraction_pt
    fraction_pt /= 100
    avg_n_nodes /= 100
    avg_n_edges /= 100



    filepath = path + 'generation/epochREEVAL'
    idx = 0
    for i_epoch in range(5, 161, 5):
        filename = filepath + str(i_epoch) + '.smi'
        with open(filename) as f:
            reader = csv.reader(f, delimiter=" ")
            data = list(zip(*reader))[0]
        data = list(data)
        while '' in data:
            data.remove('')
        while 'SMILES' in data:
            data.remove('SMILES')
        while '[Xe]' in data:
            data.remove('[Xe]')
        fraction_unique[idx] = float(len(set(data)))/len(data)
        idx += 1

    with open(path + 'generation.csv', "a") as output_file:
        for i in range(len(epoch_key)):
            output_file.write(
                f"{epoch_key[i]}, {fraction_valid[i]:.3f}, {fraction_valid_pt[i]:.3f}, {fraction_pt[i]:.3f}, "
                f"{avg_n_nodes[i]:.3f}, {avg_n_edges[i]:.3f}, {fraction_unique[i]:.3f}, "
                f"\n"
            )

if __name__ == "__main__":
    get_formal_charges_selected(path=args.path)
    print("Done.", flush=True)