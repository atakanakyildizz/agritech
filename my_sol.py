import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

# Function to read data from a CSV file and return two lists:
# one for resistance and one for watermark
def data_generator():
    t = []
    y = []
    x = []
    infile = open("calibrazione_board1.csv", "r")
    for i in infile.readlines():
        i=i.removesuffix('\n')
        i = i.removesuffix('"')
        i = i.removeprefix('"')
        t.append(i.split('\t'))
    for i in range(1,t.__len__()):
        y.append(float(t[i][0]))
        x.append(float(t[i][1]))
    return (x,y)

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
m_values= []
q_values = []

i_offset_change = [0, 11, 16, 36, 56, 76, 101]
for i in i_offset_change:
    p1 = float(pressure_values[i])
    r1 = float(resistance_values[i])
    p2 = float(pressure_values[i+1])
    r2 = float(resistance_values[i+1])
    m = (p2 - p1) / (r2 - r1)
    q = p1
    m_values.append(m)
    q_values.append(q)

# Filter the DataFrame for the plant with id 13
df = pd.read_csv('experiment.csv')
df_filtered = df.loc[df['id'] == 13]

# Collect all 'watermark' values in a list
watermark_values = df_filtered['Watermark'].tolist()

# Use the linear regression model to make predictions
res = []
intercept = model.intercept_
coef = model.coef_
w_m = []
for i in range(len(watermark_values)-1):
    w_m.append(1/float(watermark_values[i]))
for value in w_m:
    res.append(coef /value + intercept)

# Calculate potential (pot) for each result and add to a list
mat_pot = []
len_qvalues = len(q_values)
for element in res:
    for i in range(len_qvalues-2):
        if q_values[i] <= element < q_values[i + 1]:
            pot = m_values[i] * element + q_values[i]
            mat_pot.append(pot/1000)
            break
        else:
            pot = m_values[len_qvalues-1] * element + q_values[len_qvalues-1]
            mat_pot.append(pot/1000)
            break

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