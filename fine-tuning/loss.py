# load general packages and functions
import torch

# load program-specific functions
from parameters.constants import constants as C

def compute_loss(score, agent_ll, prior_ll, uniqueness_tensor):


        augmented_prior_ll = prior_ll + C.sigma*score


        difference = agent_ll - augmented_prior_ll
        loss = difference*difference

        mask = (uniqueness_tensor != 0).int()
        loss = loss * mask

        return loss