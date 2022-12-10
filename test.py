import numpy as np

path = 'data/snake/'
print(np.load(path + 'images.npy').shape)
print(np.load(path + 'joints.npy').shape)
