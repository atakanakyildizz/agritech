import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# x = watermark, y = resistance
def data_generator():
    t = []
    y = []
    x = []
    for i in open("calibrazione_board1.csv", "r").readlines():
        i = i.removesuffix('\n')
        i = i.removesuffix('"')
        i = i.removeprefix('"')
        t.append(i.split('\t'))
    for i in range(1, t.__len__()):
        y.append(float(t[i][0]))
        x.append(float(t[i][1]))
    return x, y


df = pd.read_csv('Cal_Watermark_Datasheet.csv')
pressure_values = df['Pressure'].tolist()
resistance_values = df['Resistance'].tolist()

# Örnek veri setini oluşturalım
watermark_data, resistance_data1 = data_generator()
data = {
    'Pressure': [pressure_values],
    'Resistance': [resistance_data1],
    'Watermark': [watermark_data]
}

# Veri setini DataFrame'e dönüştürelim
df = pd.DataFrame(data)

# Bağımsız değişkenler (X) ve bağımlı değişkenler (y) olarak ayıralım
X = df[['Pressure']]
y = df['Resistance']

# Verileri eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer regresyon modelini oluşturalım ve eğitelim
model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapalım
y_pred = model.predict(X_test)

# Eğitim seti ve tahminler arasındaki ilişkiyi görselleştirelim
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.title('Pressure vs Resistance')
plt.xlabel('Pressure')
plt.ylabel('Resistance')
plt.show()

# Modelin performansını değerlendirelim
print("Modelin R^2 skoru:", model.score(X_test, y_test))
