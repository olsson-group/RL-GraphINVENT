# load general packages and functions
import torch
from rdkit.Chem import MolFromSmiles, QED, Crippen, Descriptors, rdMolDescriptors, AllChem
import numpy as np
from rdkit import DataStructs

# load program-specific functions
from parameters.constants import constants as C


def compute_score(graphs, termination_tensor, validity_tensor, uniqueness_tensor, smiles, drd2_model):

    if C.score_type == "reduce":
        # Reduce size
        n_nodes = graphs[2]
        n_graphs = len(n_nodes)
        max_nodes = C.max_n_nodes
        score = torch.ones(n_graphs, device="cuda") - torch.abs(n_nodes - 10.) / (max_nodes - 10 + 1)
    
    elif C.score_type == "augment":
        # Augment size
        n_nodes = graphs[2].float()
        n_graphs = len(n_nodes)
        max_nodes = C.max_n_nodes
        score = torch.ones(n_graphs, device="cuda") - torch.abs(n_nodes - 40.) / (max_nodes - 40)
    
    elif C.score_type == "qed":
        # QED
        score = [QED.qed(MolFromSmiles(smi)) for smi in smiles]
        score = torch.tensor(score, device="cuda")


    elif C.score_type == "activity":
        n_mols = len(smiles)

        mols = [MolFromSmiles(smi) for smi in smiles]

        # QED
        qed = [QED.qed(mol) for mol in mols]
        qed = torch.tensor(qed, device="cuda")
        qedMask = torch.where(qed > 0.5, torch.ones(n_mols, device="cuda", dtype=torch.uint8), torch.zeros(n_mols, device="cuda", dtype=torch.uint8))
        
        activity = compute_activity(mols, drd2_model)
        activityMask = torch.where(activity > 0.5, torch.ones(n_mols, device="cuda", dtype=torch.uint8), torch.zeros(n_mols, device="cuda", dtype=torch.uint8))


        score = qedMask*activityMask

    else:
        raise NotImplementedError("The score type chosen is not defined. Please choose among 'reduce', 'augment', 'qed' and 'activity'.")
    
    # remove non unique molecules from the score
    score = score * uniqueness_tensor

    # remove invalid molecules
    score = score * validity_tensor

    # remove non properly terminated molecules
    score = score * termination_tensor

    return score


def compute_activity(mols, drd2_model):

    n_mols = len(mols)

    activity = torch.zeros(n_mols, device="cuda")

    for idx, mol in enumerate(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)   
        ecfp4 = np.zeros((2048,))                                       
        DataStructs.ConvertToNumpyArray(fp, ecfp4)
        activity[idx] = drd2_model.predict_proba([ecfp4])[0][1]
    
    return activity

        
def compute_activity_score(graphs, termination_tensor, validity_tensor, uniqueness_tensor, smiles):
    
    n_mols = len(smiles)

    mols = [MolFromSmiles(smi) for smi in smiles]

    # QED
    qed = [QED.qed(mol) for mol in mols]
    qed = torch.tensor(qed, device="cuda")
    qedMask = torch.where(qed > 0.5, torch.ones(n_mols, device="cuda", dtype=torch.uint8), torch.zeros(n_mols, device="cuda", dtype=torch.uint8))
    
    activity = compute_activity(mols)
    activityMask = torch.where(activity > 0.5, torch.ones(n_mols, device="cuda", dtype=torch.uint8), torch.zeros(n_mols, device="cuda", dtype=torch.uint8))


    score = qedMask*activityMask

    # remove non unique molecules from the score
    score = score * uniqueness_tensor

    # remove invalid molecules
    score = score * validity_tensor

    # remove non properly terminated molecules
    score = score * termination_tensor

    return score
    