import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import csv
import math

good_picks = [6,7,8,19,21,25,27,29,35]
bad_picks = [1,2,3,4,5,10,11,13,14,16,17,18,22,23,24,26,30,31,32,33,34,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]

deletor = np.concatenate((good_picks, bad_picks))
deletor.sort()

# for repeator in range(1):

base_aray = np.delete(range(58),deletor)

min_error_train = 0
min_error_train_seed = []
min_error_test = 0
min_error_test_seed = []
best_ac_100 = 0
best_ac_100_seed = []
best_ac_500 = 0
best_ac_500_seed = []

seed_limit = math.floor(math.pow(2, base_aray.__len__()))
seed_limit = base_aray.__len__()

for seed in range(1, seed_limit, 1):
    ### TRAIN
    with open('train.csv', 'r') as f:
        reader = csv.reader(f)
        traindataset = list(reader)

    traindataset.pop(0)

    relevant_stuff = []
    for f in good_picks:
        relevant_stuff.append(f)
    relevant_stuff.append(base_aray[seed])

    seed_copy = seed
    # for idx, val in enumerate(base_aray):
    #     if seed_copy % 2 == 1:
    #         relevant_stuff.append(val)
    #     seed_copy = math.floor(seed_copy / 2)
    #     if seed_copy == 0:
    #         break

    print("Seed:",seed)
    print("relevant stuff:",relevant_stuff)

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


    # print(np.shape(truedataset))

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
    # print("FEATURES SHAPE:", traindataset.shape)

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
    error = []

    for i in range(0, iterations, 1):
        pred = np.dot(traindataset, theta)
        error = pred - y

        cost_derivate = np.dot(error, traindataset) + np.dot(regparam, theta)
        new_theta = theta - learning_rate * cost_derivate / m

        theta = new_theta
        # if i % 1000 == 0:
        #     print('mean error:', np.abs(error).mean())

    if np.abs(error).mean() < min_error_train:
        min_error_train = np.abs(error).mean()
        min_error_train_seed = relevant_stuff

    # difference = theta - original
    # print('difference: ', difference)

    # for idx, val in enumerate(difference):
    #     if val > 600:
    #         print(idx)


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

    # print(predictions.shape)
    # print(testy.shape)
    # print(error.shape)

    for k in error:
        if k <= 500:
            accept_rate_500 += 1
            if k <= 100:
                accept_rate_100 += 1


    accept_rate_500 = accept_rate_500 / testy.__len__()
    accept_rate_100 = accept_rate_100 / testy.__len__()

    mean_error = error.mean()

    # The coefficients
    # print('Coefficients: \n', theta)
    print('mean error:', mean_error)
    print('accept rate 500:', accept_rate_500)
    print('accept rate 100:', accept_rate_100)

    if accept_rate_100 > best_ac_100:
        best_ac_100 = accept_rate_100
        best_ac_100_seed = relevant_stuff

    if accept_rate_500 > best_ac_500:
        best_ac_500 = accept_rate_500
        best_ac_500_seed = relevant_stuff

    if mean_error < min_error_test:
        min_error_test = mean_error
        min_error_test_seed = relevant_stuff

            # # Plot outputs
    # plt.scatter(range(testy.__len__()), testy,  color='black', linewidth=1)
    # plt.plot(range(predictions.__len__()), predictions, color='blue', linewidth=1)
    #
    # plt.xticks(())
    # plt.yticks(())
    #
    # plt.show()
print("best ac 100 was:",best_ac_100, " on features ",best_ac_100_seed)
print("best ac 500 was:", best_ac_500, " on features ", best_ac_500_seed)
for f in best_ac_100_seed:
    if f not in good_picks:
        print("grabbing feature ",f)
        good_picks.append(f)