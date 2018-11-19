from keras.layers import Dense, Input
from keras.models import Model
from keras.models import Sequential
import keras
from keras.layers import BatchNormalization
import keras
import os
os.environ["PATH"] += os.pathsep + 'file:///home/gautom/anaconda3/envs/pyenv35/lib/graphviz/'

model = Sequential()
model.add(BatchNormalization())
model.add(Dense(11, input_dim=11, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(11))
rms_prop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

from keras.utils import plot_model
plot_model(model, to_file='model.png')

import csv

results = []
with open("noNA.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)

def remove_col(arra, index):
    for row in arra:
        del row[index]

    return arra
results = remove_col(results, 0)
results = remove_col(results, 0)
resul = results[1:]

fina = []
for row in resul:
    rowa = []
    for val in row:
        rowa.append(float(val))
    fina.append(rowa)

print("Coefficients:")
print(results[0])
import random
random.shuffle(fina)
random.seed(128)

X_train = []
y_train = []
for row in fina:
    X_train.append(row[0:11])
    y_train.append(row[11])

import numpy as np
X_train = np.asarray(X_train)

model.fit(X_train, X_train, epochs=1500)

results = []
with open("yesNA.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)

def remove_col(arra, index):
    for row in arra:
        del row[index]

    return arra
results = remove_col(results, 0)
results = remove_col(results, 0)
resul = results[1:]

fina = []
for row in resul:
    rowa = []
    for val in row:
        if val == "NA":
            rowa.append(np.NaN)
        else:
            rowa.append(float(val))
    fina.append(rowa)

print("Coefficients:")
print(results[0])

X_train = []
y_train = []
for row in fina:
    X_train.append(row[0:11])
    y_train.append(row[11])

print(X_train[0])

X_tra = np.asarray(X_train)
print(X_tra[0].shape)

out = model.predict(X_tra)
print("____________--")
print(out.shape)
allf = out.tolist()
print(X_tra[0])

finis = []
finis.append(results[0])
count = 0
for row in allf:
    counter = 0
    rower = []
    for val in row:
        print(row)
        if resul[count][counter] == "NA":
            print('nan')
            rower.append(val)
        else:
            rower.append(resul[count][counter])
        counter += 1
    print("*"*10)
    print(rower)
    finis.append(rower)
    count += 1
import csv
with open("S_filledNA.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(finis)