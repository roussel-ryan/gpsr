import matplotlib.pyplot as plt
import numpy as np

# generate 10^6 random particles in 2D
beam = np.random.randn(1000000, 2)

fig, ax = plt.subplots()
ax.plot(*beam.T, '.')

plt.show()
