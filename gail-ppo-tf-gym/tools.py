import scipy as sp
from scipy import stats
import numpy as np


def kl_divergence(obs, Policy1, Policy2):
    epsilon = 0.00001
    acts1 = Policy1.get_action_prob(obs=obs)[0]+epsilon
    acts2 = Policy2.get_action_prob(obs=obs)[0]+epsilon

    #print(acts1, acts2)

    #kl = stats.entropy(acts1, acts2)
    kl = np.sum(acts1*np.log(acts1/acts2))

    return kl
    

