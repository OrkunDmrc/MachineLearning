import pandas as pd
import numpy as np

salalaries = { "Isim":["Ali","Mehmet","Samet","Demet"],
                "Department": ["Yazilim","Pazarlama","Satis","Yazilim"],
              "Maas":[100,245,542,547]}
salalaryDataFrame = pd.DataFrame(salalaries)
print(salalaryDataFrame)
print(salalaryDataFrame["Department"].unique())
print(salalaryDataFrame["Department"].nunique())#number of unique
print(salalaryDataFrame["Department"].value_counts())

def bruttenNete(maas):
    return maas * 0.66
print(salalaryDataFrame["Maas"].apply(bruttenNete))
print(salalaryDataFrame.isnull())

#pivot table
print("Privot Table")
newData = {"Film":["S Park","S Park", "Simpson", "Simpson", "Simpson"],
           "Name":["Carman","Tedy","Homer","Bart","Bart"],
           "Age": [10,9,50,10,20]}
newDataFrame = pd.DataFrame(newData)
print(newDataFrame.pivot_table(values="Age",index=["Film","Name"],aggfunc=np.sum))
print(newDataFrame.pivot_table(values="Age",index=["Film","Name"],aggfunc=np.average))


