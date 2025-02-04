import numpy as np
import matplotlib.pyplot as plt

npArray1 = np.linspace(0,10,20)
npArray2 = npArray1 ** 2

plt.scatter(npArray1, npArray2)
plt.show()
plt.hist(npArray2)
plt.show()
plt.boxplot(npArray1)
plt.show()