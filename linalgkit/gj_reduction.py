"""
Here we introduce the routines and methods we have been building and studying.
Every single function that we have used in the notebooks can be found here.
Essentially, the implementation of these functions do not differ at all from 
what we explained there, so do not be surprised to see the same code twice.

For convinience, I have introduced every funcion in the same order as we introduce
them in the noteobooks.

Finally, I hope this to be useful to anybody concerned with implementing linear 
algebra algorythms: even if it just serves as slight inspiration, I am okay with
that. 

Greatings, 
puvvlo
"""

######################################
## 1. Gauss-Jordan Method for matrices
######################################

import numpy as np

def gj_reduction(matrix: np.ndarray, tol = 1e-12, give_syspar: bool=False ) -> any:
    '''
    Uses Gauss-Jordan reduction method to reduce a given matrix of any shape. 
    If specified, it also gives parameters of the system associated with the matrix:
    number of unknows and stairs.

    Parameters
    ----------
    `matrix`: array_like
        A 2D array represeting the matrix we want to reduce
    
    `give_syspar`: bool
        If set to True, it gives the parameters of the system associated
        with the given matrix: number of unknows and stairs of the reduced
        augmented and coefficient matrices.

    `tol`: float
        It sets the tolerance value by wich the elements are compared: every 
        value below `tol` is set to zero to avoid data precision issues.

    Returns
    -------
    A tuple with the following objects:
    
    `A`: array_like
        Final reduced matrix.
    
    `unknowns`: int
        The number of columns of the matrix minus one. This would be the number
        of unknowns of the corresponding system.
        
    `aug_s`, `coef_s`: int
        If the given matrix corresponds to a system of linear equations, `aug_s` 
        and `coef_s` are the number of stairs of the augmented and coefficient
        matrices, respectively.

    Notes
    -----
    - The function can be used to only reduce a given matrix. It is not compulsory
    for the matrix to be associated with a system of linear equations.

    - In case the matrix corresponds to a system of equations, the very last 
    column of the given matrix is interpreted as the constant/independent values. 
    That is why two number of stairs are given, and why the number of unknows is the
    number of columns minus one.

    - The number of stairs of the augmented and coefficient matrices are also
    identified as the ranges of both matrices. They are used to study the compatibiliy
    of the system.
            
    '''

    # Make a copy of the matrix and changes dtype
    A = matrix.astype(float).copy()
    m, n = matrix.shape 

    #We start reducing the matrix
    k, l = 0, 0     # Row and column indices
    while (k <= m-1) and (l <= n-1):
        
        # 0. Set to zero every single value below tolerance
        A[A < tol] = 0

        # 1. Prepare the rows by switching/pivoting accordingly

        col = A[k:, l]
        row = A[k, l:]

        # We find the nonzero values along that column
        nonzero_idx = col.nonzero()[0]

        # Check if the column is full of zeros
        if nonzero_idx.size == 0: 
            # Move to the next column if so and skip
            l += 1  
            continue
        
        # Find the min value along that column (minimum criteria for pivoting)
        min_nonzero_idx = nonzero_idx[np.argmin(np.abs(col)[np.abs(col) != 0])] + k

        # Switch rows
        A[[k, min_nonzero_idx], l:] = A[[min_nonzero_idx, k], l:]

        # Make a 1 in the row
        A[k, l:] = row / row[0]

        # 2. Perform operations

        # Rows that do not match with the selected row
        idx = np.arange(m)
        idx = idx[idx != k]
        
        # Block build with previous rows and first coefficients 
        block = A[idx, l:]
        coef = block[:, [0]]

        # Row operations
        A[idx, l:] = block - coef * row  
        
        # 3. Move to the next iteration
        
        k += 1
        l += 1
    
    # Save reduced matrix
    results = [A]

    # Give parameters of the system: unknows and stairs
    if give_syspar:
        aug_s = np.unique(A.nonzero()[0]).size  # Augmented matrix
        coef_s = np.unique(A[:, :-1].nonzero()[0]).size # Coefficient matrix
        results.extend([n-1, aug_s, coef_s])

    # Return only reduced matrix if no extra info requested, else tuple
    if len(results) == 1:
        return A
    else:
        return tuple(results)