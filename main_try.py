import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as mt

data = pd.read_csv("Cal_Watermark_Datasheet.csv")


Pressure = data["Pressure"].values.reshape(-1, 1)
Resistance = data["Resistance"].values.reshape(-1, 1)


reg = lm.LinearRegression()
x_train, x_test, y_train, y_test = ms.train_test_split(Pressure, Resistance, test_size=1/3, random_state=0)

reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

score = mt.r2_score(y_test, y_pred)

# Blue line is predicted and red line is our results
plt.scatter(Pressure, Resistance, color="r")
plt.plot(x_test, y_pred, color="b")
plt.show()

result = dict()
for i in range(len(y_pred)-1):
    result[int(x_test[i])] = float(y_pred[i])

sorted_dict = dict(sorted(result.items()))

for keys, values in sorted_dict.items():
    print(keys, "-->", values)
print("Score:", score)
