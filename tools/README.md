# Tools
This directory contains various tools for analyzing datasets and the generated molecules:

* [analyse_qed_size.py](./analyse_qed_size.py) : Generates a plot of the QED values of the molecules in a file against their number of nodes.
* [atom_types.py](./atom_types.py) : Gets the atom types present in a set of molecules.
* [combine_HDFs.py](./combine_HDFs.py) : Combines multiple sets of preprocessed HDF files into one set (useful when preprocessing large datasets in batches).
* [compute_intersection.py](./compute_intersection.py) : Computes the intersection between two .smi files.
* [distr_n_nodes.py](./distr_n_nodes.py): Gets the distribution of the number of nodes per molecule present in a set of molecules.
* [formal_charges.py](./formal_charges.py) : Gets the formal charges present in a set of molecules.
* [max_n_nodes.py](./max_n_nodes.py): Gets the maximum number of nodes per molecule in a set of molecules.
* [recompute_stats.py](./recompute_stats.py): Recomputes the statistics when there are more than 1 generation batch..
* [remove_groups.py](./remove_groups.py) : Creates a new dataset which does not have entries corresponding to groups of molecules.
* [remove_large_mols.py](./remove_large_mols.py) : Creates a new dataset which only includes molecules smaller than the specified percentile.
* [score_mols.py](./score_mols.py) : Computes the mean QED, mean DRD2 activity, fraction of active and unique mols, fraction unique, and average activity score for the molecules in the file.
* [select_atom_types.py](./select_atom_types.py) : Selects a set of molecules with the desired atom types.
* [select_formal_charges.py](./select_formal_charges.py) : Selects a set of molecules with the desired formal charges.


To use any tool in this directory, first activate the RL-GraphINVENT virtual environment, then run:

```
(RL-GraphINVENT-env)$ python {script} --smi path/to/file.smi
```

Simply replace *{script}* by the name of the script e.g. *max_n_nodes.py*, and *path/to/file* with the name of the SMILES file to analyze.
