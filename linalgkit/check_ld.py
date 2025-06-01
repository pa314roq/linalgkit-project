#######################################################
## 2. Check linear dependence between a set of vectors
#######################################################

import numpy as np
from . import gj_reduction

def check_ld(*vectors_set: np.ndarray) -> str:
    """
    Tells you if the set of vectors (1D arrays) is linearly dependent or
    independent.

    Parameters
    ----------
    `vectors_set`: array_like
        1D arrays with the set of vectors we want to check.

    Returns
    -------
    `str`: A string message telling you if it is linearly independent or not.
    """

    # 1. We first merge the vectors to generate the matrix we want to reduce

    n = len(vectors_set)
    m = len(vectors_set[0])
    matrix = np.zeros((m, n+1))
    for k, v in enumerate(vectors_set):
        matrix[:, k] = v

    # 2. Reduce the matrix by GJ

    A, unknows, stairs, s = gj_reduction(matrix)

    # 3. Study solution

    if unknows == stairs:
        return print("Linearly independent set")
    else:
        return print("Linearly dependent set")