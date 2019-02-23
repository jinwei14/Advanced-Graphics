from matplotlib import pyplot as mp
import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power((x-0.9) - mu, 2.) / (2 * np.power(sig, 2.)))-0.8824

for mu, sig in [(0, 1)]:
    mp.plot(gaussian(0, mu, sig))

print gaussian(0.5, 0, 1)