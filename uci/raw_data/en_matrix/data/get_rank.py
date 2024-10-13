import numpy as np
from numpy.linalg import matrix_rank 
data = np.loadtxt("data.txt")
print(data.shape)
print(matrix_rank(data))
