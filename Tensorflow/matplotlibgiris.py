import numpy as np
import matplotlib.pyplot as plt

ageList = np.sort(np.random.randint(10,50,10))
weightList = np.sort(np.random.randint(10,70,10))
#plt.plot(ageList,weightList,"g")
#plt.xlabel("ages")
#plt.ylabel("weights")
#plt.title("Age and weight")
##plt.show()
#
npArray1 = np.linspace(0,10,20)
npArray2 = npArray1 ** 3
#plt.subplot(1,2,1)
#plt.plot(npArray1, npArray2, "r--")#b*-
#plt.subplot(1,2,2)
#plt.plot(npArray2, npArray1, "b*-")#b*-
#plt.show()

myFigure = plt.figure()
figureAxes1 = myFigure.add_axes([0.1,0.1,0.5,0.5])
figureAxes2 = myFigure.add_axes([0.2,0.3,0.2,0.2])

figureAxes1.plot(npArray1,npArray2,"g--")
figureAxes1.set_xlabel("X")
figureAxes1.set_ylabel("Y")
figureAxes1.set_title("Grafik")

figureAxes2.plot(npArray2, npArray1, "r--")
figureAxes2.set_xlabel("X")
figureAxes2.set_ylabel("Y")
figureAxes2.set_title("Kücük")
plt.show()

(myFigure, myAxes) = plt.subplots(nrows=1, ncols=2)
for axes in myAxes:
    axes.plot(npArray1,npArray2,"g")
    axes.set_xlabel("X")
myFigure.tight_layout()
plt.show()

myFigure = plt.figure(dpi=100)#dpi çözünürlük figsize=(6,6),
myAxes = myFigure.add_axes([0.2,0.2,0.6,0.6])
myAxes.plot(npArray1, npArray1 ** 2, label = "üst 2")
myAxes.plot(npArray1, npArray1 ** 3, label = "üst 3")
myAxes.legend(loc=6)
myFigure.savefig("benimfigure.png", dpi=150)
plt.show()
