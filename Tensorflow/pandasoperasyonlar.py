import pandas as pd
import numpy as np

data = {"istanbul": [10,11,np.nan], "Ankara": [5,np.nan,7],"Izmir": [20,17,15],"Antalya":[20,np.nan,np.nan]}
weather = pd.DataFrame(data)
print(weather) 
print(weather.dropna())#nan değerleri düşürür
print(weather.dropna(axis=1))#columnda nan  değerleri düşürür
print(weather.dropna(axis=1, thresh=2))#columnda iki veya daha fazla nan değerleri düşürür
print(weather.fillna(20))#nan ları 20 yapar

#group by
salalaries = {"Department": ["Yazilim","Yazilim","Satis","Satis","Hukuk","Hukuk"],
              "Isim":["Ali","Mehmet","Samet","Demet","Sacid","Atil"],
              "Maas":[100,245,542,547,654,123]}
salalariesDataFrame = pd.DataFrame(salalaries)
grouped = salalariesDataFrame.groupby("Department")
print(grouped.count())
print(grouped.mean("Maas"))#grupsal ortalama
print(grouped.min())
print(grouped.max())
print(grouped.describe())#genel hesaplama
