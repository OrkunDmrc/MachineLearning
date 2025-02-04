import numpy as np
import pandas as pd
#series
myDictionary = {"Atil": 50, "Zeynep": 40, "Mehmet": 30}
print(pd.Series(myDictionary))
print(pd.Series([50,40,30], ["Atil", "Zeynep", "Mehmet"]))
print(pd.Series(data = [50,40,30], index = ["Atil", "Zeynep", "Mehmet"]))
competitors = ["Atil", "Zeynep", "Mehmet"]
result1 = pd.Series([10,5,1],competitors)
resutl2 = pd.Series([10,5,1],competitors)
print(result1 + resutl2)
