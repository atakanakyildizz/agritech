import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

# Function to read data from a CSV file and return two lists:
# one for resistance and one for watermark
def data_generator():
    x = []
    y = []
    infile = open("calibrazione_board1.csv", "r")
    for i in infile.readlines():
        i = i.removeprefix('"')
        i = i.removesuffix('\n')
        i = i.removesuffix('"')
        i = i.split('\t')
        x.append((i[0]))
        y.append(i[1])
    x.pop(0), y.pop(0)
    for i in range(0, len(x)-1):
        x[i] = float(x[i])
        y[i]= float(y[i])
    return(x, y)


 #   a,b = infile.readline().split()
 #   while a != "end" and b != "end":
 #       a, b = infile.readline().split()
 #       if a == "end":
 #           break
 #       x.append(float(b)), y.append(float(a))


# Function to fit a linear regression model on the data
def fit_linear_regression(a, b):
    t = []  #resistance
    t_ = [] #watermark
    for i in range(len(a)):
        if (a[i]) < 2000:
            t.append(1/a[i])
            t_.append(b[i])
    x = np.array(t).reshape((-1,1))
    y = np.array(t_)
    model = LinearRegression().fit(x, y)
    return model

def main():
    # Read resistance and pressure values from a datasheet
    Datasheet = pd.read_csv('Cal_Watermark_Datasheet.csv')
    pressure_values = Datasheet['Pressure'].tolist()
    resistance_values = Datasheet['Resistance'].tolist()

    # Perform linear regression on the watermark and resistance values
    resistance, watermark = data_generator()
    model = fit_linear_regression(resistance, watermark)

    # Calculate the slope (m) and intercept (q) for each pair of values (the offset changes)
    res_for_reg = []
    for i in resistance_values:
        res_for_reg.append(float(i))
    res_for_reg1 = np.array(res_for_reg).reshape((-1,1))
    model1 = LinearRegression().fit(res_for_reg1,pressure_values)
    M = model1.coef_
    Q = model1.intercept_

    # Filter the DataFrame for the plant with id 13
    df = pd.read_csv('experiment.csv')
    df_filtered = df.loc[df['id'] == 13]

    # Collect all 'watermark' values in a list
    watermark_values = df_filtered['Watermark'].tolist()

    # Use the linear regression model to make predictions
    res = []
    intercept = model.intercept_
    coef = model.coef_

    print(intercept)
    print(coef)
    w_m = []
    for i in range(len(watermark_values)-1):
        w_m.append(1/float(watermark_values[i]))
    for value in w_m:
        res.append(coef /value + intercept)
    mat_pot = []
    for i in res:
        mat_pot.append(M*i + Q)

    # Calculate potential (pot) for each result and add to a list


    # Create a list of timestamps (crono)
    lenght = len(mat_pot)
    crono = []
    for i in range(lenght):
        crono.append(i)

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract dates for id = 13
    dates_id_13 = df.loc[df['id'] == 13, 'Date']

    # Plot the results
    plt.figure()
    plt.plot(crono, mat_pot)
    plt.show()

if __name__ == '__main__':
    main()
