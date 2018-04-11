import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import csv

### TRAIN
with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    traindataset = list(reader)

traindataset.pop(0)
y = []

leng = 0

# Treat dataset by removing first and second features (non-preditctors)
# Also build target vector (last feature of every entry)
for entry in traindataset:
    y.append(entry.pop(60))
    entry.pop(0)
    entry[0] = 0
    leng = entry.__len__()

max = np.zeros(leng)
min = np.zeros(leng)

traindataset = np.array(traindataset).astype(float)
print(traindataset.shape)

for entry in traindataset:
    for f in range(0, entry.__len__(), 1):
        entryf = float(entry[f])
        if entryf > max[f]:
            max[f] = entryf
        if entryf < min[f]:
            min[f] = entryf

divisor = max - min

for entry in traindataset:
    for f in range(0, entry.__len__(), 1):
        if divisor[f] != 0:
            entry[f] = (entry[f] - min[f]) / divisor[f]


regression = linear_model.LinearRegression()
regression.fit(traindataset, y)



### TEST
with open('test.csv', 'r') as f:
    reader = csv.reader(f)
    testdataset = list(reader)

with open('test_target.csv', 'r') as f:
    reader = csv.reader(f)
    testy = list(reader)

testdataset.pop(0)
testy.pop(0)

# Treat dataset by removing first and second features (non-preditctors)
# Also build target vector (last feature of every entry)
for entry in testdataset:
    entry.pop(0)
    entry[0] = 0

testy = np.array(testy).astype(float)
testdataset = np.array(testdataset).astype(float)

for entry in testdataset:
    for f in range(0, entry.__len__(), 1):
        if divisor[f] != 0:
            entry[f] = (entry[f] - min[f]) / divisor[f]

predictions = np.array(regression.predict(testdataset)).astype(float)

print(predictions.shape)

print(testdataset.shape)
print(testy.shape)

# The coefficients
print('Coefficients: \n', regression.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(testy, predictions))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(testy, predictions))

error = abs(predictions - testy.T)
error = error.T
print(error.shape)
ac_500 = 0
ac_100 = 0
for e in error:
    if e <= 500:
        ac_500 += 1
        if e <= 100:
            ac_100 += 1

ac_100 = ac_100 / testy.__len__()
ac_500 = ac_500 / testy.__len__()
print("error: ", error.mean())
print("ac 500:", ac_500)
print("ac 100:", ac_100)
#
# # Plot outputs
# plt.scatter(range(testy.__len__()), testy,  color='black', linewidth=1)
# plt.plot(range(predictions.__len__()), predictions, color='blue', linewidth=1)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()