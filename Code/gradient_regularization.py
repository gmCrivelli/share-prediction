import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import csv

### TRAIN
with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    traindataset = list(reader)

relevant_stuff = [1,2,6,8,11,18,19,20,22,23,25,26,27,28,29,41,43,47,48]
#relevant_stuff = range(0,59)

traindataset.pop(0)


y = []

max = np.zeros(relevant_stuff.__len__())
min = np.zeros(relevant_stuff.__len__())

truedataset = []

# Treat dataset by removing first and second features (non-preditctors)
# Also build target vector (last feature of every entry)
for idx, entry in enumerate(traindataset):
    if float(entry[4]) > 0:
        y.append(entry.pop(60))
        truedataset.append(entry)
        entry.pop(0)
        entry[0] = 0.0


print(np.shape(truedataset))

traindataset = truedataset
traindataset = np.array(traindataset).astype(float)
traindataset = traindataset.take(relevant_stuff, axis=1)

m = traindataset.__len__()

for entry in traindataset:
    for f in range(0, entry.__len__(), 1):
        entryf = float(entry[f])
        if entryf > max[f]:
            max[f] = entryf
        if entryf < min[f]:
            min[f] = entryf


divisor = max - min
truedataset = []


for entry in traindataset:
    for f, val in enumerate(entry):
        if divisor[f] != 0:
            entry[f] = (entry[f] - min[f]) / divisor[f]
    squared = entry * entry
    new_entry = np.concatenate((entry, squared))
    truedataset.append(new_entry)

truedataset = np.array(truedataset).astype(float)
traindataset = truedataset
print("FEATURES SHAPE:", traindataset.shape)

y = np.array(y).astype(float)

theta = np.zeros(2 * relevant_stuff.__len__())
regparam = np.zeros(2 * relevant_stuff.__len__())

regparam += 1000
regparam[0] = 0

for i in range(0,relevant_stuff.__len__(),1):
   theta[i] = random.uniform(-1000,1000)

new_theta = theta
iterations = 10000
learning_rate = 0.05

original = theta
costs = []

for i in range(0, iterations, 1):
    pred = np.dot(traindataset, theta)
    error = pred - y

    cost_derivate = np.dot(error, traindataset) + np.dot(regparam, theta)
    new_theta = theta - learning_rate * cost_derivate / m

    theta = new_theta
    meanerr = np.abs(error).mean()
    costs.append(meanerr)
    if i % 100 == 0:
        print('mean error:', np.abs(error).mean())

difference = theta - original
print('difference: ', difference)

for idx, val in enumerate(difference):
    if val > 600:
        print(idx)


### TEST
with open('test.csv', 'r') as f:
    reader = csv.reader(f)
    testdataset = list(reader)

with open('test_target.csv', 'r') as f:
    reader = csv.reader(f)
    testy = list(reader)

testdataset.pop(0)

testy.pop(0)

predictions = []

# Treat dataset by removing first and second features (non-preditctors)
# Also build target vector (last feature of every entry)
for entry in testdataset:
    entry.pop(0)
    entry[0] = 0.0

testdataset = np.array(testdataset).astype(float)
testdataset = testdataset.take(relevant_stuff, axis=1)

truetest = []

for entry in testdataset:
    e = np.array(entry).astype(float)
    for f in range(0, entry.__len__(), 1):
        if divisor[f] != 0:
            e[f] = (e[f] - min[f]) / divisor[f]
    squared = e * e
    new_e = np.concatenate((e, squared))
    truetest.append(new_e)

truetest = np.array(truetest).astype(float)
for entry in truetest:
    predictions.append(np.dot(entry, theta))

predictions = np.array(predictions).astype(float)
testy = np.array(testy).astype(float)

error = predictions - testy.T

error = error.T

error = np.abs(error)

threshold = 100

accept_rate_100 = 0
accept_rate_500 = 0

print(predictions.shape)
print(testy.shape)
print(error.shape)

for k in error:
    if k <= 500:
        accept_rate_500 += 1
        if k <= 100:
            accept_rate_100 += 1


accept_rate_500 = accept_rate_500 / testy.__len__()
accept_rate_100 = accept_rate_100 / testy.__len__()

# The coefficients
print('Coefficients: \n', theta)
print('mean error:', error.mean())
print('accept rate 500:', accept_rate_500)
print('accept rate 100:', accept_rate_100)

# Plot outputs
#plt.scatter(range(testy.__len__()), testy,  color='black', linewidth=1)
plt.plot(range(iterations), costs, color='blue', linewidth=1)
plt.xlabel("Iterations")
plt.ylabel("Mean error (in shares)")
plt.title("GD + Regularization")
plt.show()