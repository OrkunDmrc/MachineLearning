import numpy as np

myArray = np.arange(0,15)
print(myArray)
myArray[3:8] = -1
print(myArray)
#reference stack kaçınma
copyArray = myArray.copy()
print(copyArray)
slicingArray = copyArray[0:5]
slicingArray[:] = 0
print(slicingArray)
print(myArray)
#matrix
print("matix")
myArray = np.array([[10,20,30],[40,50,60],[70,80,90]])
print(myArray)
print(myArray[0:2,2])

myNewArray = np.arange(0,25)
myNewArray = myNewArray.reshape(5,5)
print(myNewArray)
print(myNewArray[[4,0,2]])
#operasyonlar
print("operasyonlar")
myArray = np.random.randint(0,100,20)
newMyArray = myArray > 20
print(newMyArray)
print(myArray[newMyArray])
print(myArray[myArray > 24])
myArray = np.arange(0,24)
print(myArray + myArray)
print(myArray * myArray)
print(np.sqrt(myArray))
