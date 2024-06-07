import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

# Function to read data from a CSV file and return two lists:
# one for resistance and one for watermark
def data_generator():
    t, y, x = [], [], []
    for i in open("calibrazione_board1.csv", "r").readlines():
        i = i.removesuffix('\n')
        i = i.removesuffix('"')
        i = i.removeprefix('"')
        t.append(i.split('\t'))
    for i in range(1, t.__len__()):
        y.append(float(t[i][0])) #ohm unit
        x.append(float(t[i][1])/1000) #mm unit
    return x,y

# Function to fit a linear regression model on the data
def fit_linear_regression(a, b):
    t = []  #resistance
    t_ = [] #watermark
    for i in range(len(a)):
        if a[i] < 2000:
            t.append(1/a[i])
            t_.append(b[i])
    x = np.array(t).reshape((-1,1))
    y = np.array(t_)
    model = LinearRegression().fit(x, y)
    return model

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
