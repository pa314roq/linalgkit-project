####################################
## 3. General solution for a system
####################################

import numpy as np
from . import gj_reduction

def solve(coef_matrix: np.ndarray, values: np.ndarray=None) -> tuple:
    """ 
    Solves for the general solution of a linear system of equations with coefficient matrix 
    `coef_matrix` and independent values `values`, with GJ reduction method: i.e. gives the
    solution of the system as a combination of a particular solution `part_sol` of the 
    nonhomogeneous system, and the indepedent solutions `indp_sol` of the corresponding 
    homogeneous system.

    Parameters
    ----------
    `coef_matrix`: array_like
        2D array with the coefficients matrix of the system.
    
    `values`: array_like
        1D array with the independent values of the system.

    Returns
    -------
    A tuple with the following objects:
    
    `part_sol`: array_like
        Array of a particular solution of the system
    
    `indp_sol`: list of arrays
        A list with the independent solutions of the corresponding homogeneous system

    Notes
    -----
    1. The functions ditinguish between all posible cases:

    - If the system is uncompatible, it raises a warning that the system has no solution.
    - If the system is determined, it just gives the particular solution.
    - If the system is underdetermined, it gives all the solutions.

    2. The function also resolves homogeneous systems. In this case, just give the coefficient
    matrix (no `values` needed). If the system is determined, it gives a null solution; if 
    underdetermined, it just gives the independent solutions.
    """
    
    # Check if we have a homogeneous system
    if values is None:
        val = np.zeros(coef_matrix.shape[0])
    else:
        val = values.copy()
    
    #Create the augmented matrix
    matrix = np.hstack((coef_matrix, val[:, np.newaxis]))

    # Reduce matrix of the system
    A, n, aug_s, coef_s = gj_reduction(matrix, give_syspar=True)

    # Check if system has no solution
    if aug_s != coef_s:
        return print("Incompatible system: no solution")
    
    # 1. Identify independent and dependent variables

    # Indices for nonfree variables (in the stairs)
    stairs_idx = np.array([
        np.argmax(row != 0)
        for row in A[:coef_s, :-1]
    ])

    #Indices for free variables (not in the stairs)
    nostairs_idx = np.array([k for k in range(n) if k not in stairs_idx])
    
    # 2. Give particular solution (null free variables)

    part_sol = np.zeros(n)
    val = A[:, -1]
    part_sol[stairs_idx] = val[:aug_s]   # Assign values to variables
    
    # 3. Give independent solutions (null free variables except one)

    indp_sol = []
    for free_idx in nostairs_idx:
        sol = np.zeros(n)
        val = - A[:, free_idx]   # Move free variable to the other side
        sol[stairs_idx] = val[:aug_s]    # Assign values to variables
        sol[free_idx] = 1

        indp_sol.append(sol)

    # Check if the system is determined
    if (aug_s == coef_s) & (coef_s == n):
        # Just give part_sol in this case
        print("Compatible determined system")
        return part_sol
    
    # Last case (underdetermined)
    else:
        print("Compatible underdetermined system")
        if np.any(values):  # Give everything if inhomgeneous
            return part_sol, tuple(indp_sol)
        else:   # Just give indp_sol if homogeneous
            return tuple(indp_sol)