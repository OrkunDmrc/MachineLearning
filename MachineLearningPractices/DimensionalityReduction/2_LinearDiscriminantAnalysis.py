from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784",version=1)

X = mnist.data
y = mnist.target.astype(int)

X_lda = LinearDiscriminantAnalysis(X,y)

plt.figure()
plt.scatter(X_lda[:,0],X_lda[:,1],X_lda[:,2],c=y,cmap="tab10",alpha = 0.6)
plt.title("LDA of MNIST Dataset")
plt.xlabel("LDA1")
plt.ylabel("LDA2")
plt.colorbar(label="Digits")
plt.show()
