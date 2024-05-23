# perform 3D reconstruction using PnP
import numpy as np
def find_matching_rows(matrix1, matrix2):
    matching_indices = []
    matching_indices_2 = []
    for i, row1 in enumerate(matrix1):
        for j, row2 in enumerate(matrix2):
            if np.array_equal(row1, row2):
                matching_indices.append(i)
                matching_indices_2.append(j)
                break
    return matching_indices,matching_indices_2