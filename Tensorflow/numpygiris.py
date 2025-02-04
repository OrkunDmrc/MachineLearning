import numpy as np
#numpy arrays
myList = [10,20,30]
np.array(myList)#numpy cinsi array oluşturma
print(np.arange(0,10))
print(np.arange(0,10,2))
print(np.zeros(2))
print(np.zeros((2,2)))
print(np.ones(2))
#linspace : arasında eşit farkla dizi oluştur
print(np.linspace(0,20,6))
print(np.linspace(0,10,20))
#eye : 
print(np.eye(3))
#random
print("random")
print(np.random.randint(1,10))#1 10 arası random
print(np.random.randint(1,10,5))#1 10 arası 5 random
print(np.random.randn(8))
print(np.random.randn(2,2))
#numpy dizi methodları
print("numpy dizi methodları")
myNumpyArray = np.random.randint(1,100,20)
print(myNumpyArray.reshape(4,5))
print(myNumpyArray.min())
print(myNumpyArray.max())
print(myNumpyArray.argmin())
print(myNumpyArray.argmax())
print(myNumpyArray.shape)