import numpy as np
import pandas as pd
data = np.random.randn(4,3)
dataFrame = pd.DataFrame(data)
print(dataFrame)
newDataFrame = pd.DataFrame(data, index=["Atil","Zeynep","Osman","Mehmet"], columns=["maas","yas","saat"])
print(newDataFrame)
print(newDataFrame["yas"])
print(newDataFrame[["yas", "maas"]])
print(newDataFrame.loc["Mehmet"])
print(newDataFrame["yas"].loc["Mehmet"])
newDataFrame["emeklilik"] = 2 * newDataFrame["yas"]
print(newDataFrame)
newDataFrame.drop("emeklilik", axis = 1, inplace=True)
print(newDataFrame)
newDataFrame.drop("Mehmet",inplace=True)
print(newDataFrame)
newDataFrame = newDataFrame[newDataFrame["yas"] > 0]
print(newDataFrame)
newDataFrame = pd.DataFrame(data, index=["Atil","Zeynep","Osman","Mehmet"], columns=["maas","yas","saat"])
newDataFrame["new index"] = ["Ati","Zey","Osm", "Meh"]
newDataFrame.set_index("new index", inplace=True)
print(newDataFrame)
print(newDataFrame.loc["Ati"])
#multi index
print("multi index")
firstIndexes = ["Simpson","Simpson","Simpson","S park","S park","S park"]
secondIndexes = ["Homer", "Lisa", "Bart", "Cartman", "Kenny", "Kyle"]
commitedIndexes = list(zip(firstIndexes, secondIndexes))
print(commitedIndexes)
commitedIndexes = pd.MultiIndex.from_tuples(commitedIndexes)
print(commitedIndexes)
myCartonChars = [[10,"A"],[20,"B"],[30,"C"],[40,"D"],[50,"E"],[60,"F"]]
myCartonChars = pd.DataFrame(myCartonChars, index=commitedIndexes, columns=["yas","meslek"])
myCartonChars.index.names = ["Film","Char"]
print(myCartonChars)
print(myCartonChars.loc["Simpson"])
print(myCartonChars.loc["Simpson"].loc["Lisa"])


