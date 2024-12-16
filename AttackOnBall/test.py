import numpy as np

a = np.arange(30).reshape((2, 5, 3))
print(a)
print(a[:, :, 0])
print(np.transpose(a).shape)