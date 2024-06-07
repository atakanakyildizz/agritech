# 1, 5, 12, 13 14 ottimali
# 6, 7, 8, 9, 10 senza acqua
# 2, 3, 4, 11, 15 con acqua ma con un patogeno (valutare parametrici statistici)
# rimaneggiare i dati di experiment.csv, sensore light non c' è
# regressione
# federico.cum@polito.it

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd


# raccolgo i dati dal file csv e li metto in due liste, una per la resistenza e una per il watermark
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


def main():
    # faccio regressione lineare tra i valori di watermark e resistenza <= 2000
    a, b = data_generator()
    t = []  # resistance
    t_ = []  # watermark
    for i in range(len(a)):
        if a[i] < 2000:
            t.append((a[i]))
            t_.append(1 / b[i])
    x = np.array(t).reshape((-1, 1))
    y = np.array(t_)
    x_ = np.array(a).reshape((-1, 1))
    y_ = np.array(b)
    model = LinearRegression().fit(x, y)

    # plt.plot(x_, y_)
    # plt.plot(x_, 1/y_new)
    # plt.show()

    # leggo dal datasheet i valori di resistenza e pressione
    df = pd.read_csv('Cal_Watermark_Datasheet.csv')
    pressure_values = df['Pressure'].tolist()
    resistance_values = df['Resistance'].tolist()

    # indici di offset per ogni coppia di valori utile
    index = [0, 11, 16, 36, 56, 76, 101]

    # calcolo il coefficente m per una coppia di valori per ogni diverso offset
    m_values = []
    q_values = []

    for i in index:
        p1 = float(pressure_values[i])
        r1 = float(resistance_values[i])
        p2 = float(pressure_values[i + 1])
        r2 = float(resistance_values[i + 1])
        m = (p2 - p1) / (r2 - r1)
        q = p1
        m_values.append(m)
        q_values.append(q)

    print(q_values)
    df = pd.read_csv('experiment.csv')
    # Filtra il DataFrame per la pianta con id 13
    df_filtered = df.loc[df['id'] == 13]

    # Raccogli tutti i valori di 'watermark' in una lista
    watermark_values = df_filtered['Watermark'].tolist()

    res = []
    a = model.intercept_
    b = model.coef_

    for element in watermark_values:
        res.append(a / element + b)

    mat_pot = []

    len_qvalues = len(q_values)
    for element in res:
        for i in range(len_qvalues - 1):
            if q_values[i] <= element < q_values[i + 1]:
                pot = m_values[i] * element + q_values[i]
                mat_pot.append(pot*1000000000)
                break
            else:
                pot = m_values[len_qvalues - 1] * element + q_values[len_qvalues - 1]
                mat_pot.append(pot*10000000000)
                break


    crono = []
    for i in range(len(mat_pot)):
        crono.append(i)

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract dates for id = 13
    dates_id_13 = df.loc[df['id'] == 13, 'Date']

    # print(dates_id_13)
    plt.figure()
    plt.plot(crono, mat_pot)
    plt.show()


if __name__ == '__main__':
    main()