import json
import numpy as np

import logging
import subprocess

from itertools import product

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    

def get_hyperparameters(jobid, *args):
    """
    Generate all combinations of parameters and return the combination corresponding to the job ID.

    This function accepts a variable number of arguments. Each argument can be a list of values for a particular parameter or a single number. In the latter case, it will be treated as a list with one element.

    The function generates all combinations of these parameters using the Cartesian product. The combinations are zero-indexed, and the combination corresponding to the job ID is returned.

    """

    args = [[arg] if not isinstance(arg, list) else arg for arg in args]
    all_combinations = list(product(*args))

    if jobid <= len(all_combinations):
        return all_combinations[jobid - 1]
    else:
        raise ValueError("Job ID is out of range.")
    


def log_info(message):
    logging.info(message)