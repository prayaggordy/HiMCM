from keras.models import Sequential
import keras
from keras.layers import BatchNormalization
import csv
import os
os.environ["PATH"] += os.pathsep + 'file:///home/gautom/anaconda3/envs/pyenv35/lib/graphviz/'

results = []
with open("noNA.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)

resultsTwo = []
with open("S_filledNA.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        resultsTwo.append(row)

def remove_col(arra, index):
    for row in arra:
        del row[index]

    return arra
results = remove_col(results, 0)
results = remove_col(results, 0)

resul = results[1:]
susa = resultsTwo[1:]
for row in susa:
    resul.append(row)

print(len(resul))

fina = []
for row in resul:
    rowa = []
    for val in row:
        print(resul.index(row))
        print(row)
        rowa.append(float(val))
    fina.append(rowa)

print("Coefficients:")
print(results[0])
import random
random.shuffle(fina)
random.seed(128)

res = fina[1:]
test = fina[1:50]
X_train = []
y_train = []
for row in res:
    print(row)
    X_train.append(row[0:11])
    y_train.append(row[11])

X_test = []
y_test = []
for row in test:
    X_test.append(row[0:11])
    y_test.append(row[11])

from keras.layers import Dense

#y_train = [val/100.0 for val in y_train]
#y_test = [val/100.0 for val in y_test]
print(y_train)
print(y_test)
# create model
model = Sequential()
model.add(BatchNormalization())
model.add(Dense(12, input_dim=12, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))
rms_prop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=rms_prop, metrics=['mae'])


from keras.utils import plot_model
plot_model(model, to_file='deep_model.png')


import numpy as np
X_test = np.asarray(X_test)
print(X_test.shape)
y_test = np.asarray(y_test)

X_train = np.asarray(X_train)
print(X_train.shape)
y_train = np.asarray(y_train)

history = model.fit(X_train, y_train, epochs=1000, batch_size=100, validation_split=0.1)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("MODEL SAVED")
def plot_history(history):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
    plt.legend()
    plt.show()

plot_history(history)

from sklearn import metrics
import numpy as np
from scipy.signal import savgol_filter
# Predict and measure RMSE
pred = model.predict(X_test)

score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Score (RMSE): {}".format(score))

def chart_regression(pred,y,sort=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    t = pd.DataFrame({'pred' : pred, 'y' : y.flatten()})
    yhat = savgol_filter(y, 49, 12)
    if sort:
        t.sort_values(by=['y'],inplace=True)
    a = plt.plot(t['y'].tolist(),label='expected')
    b = plt.plot(t['pred'].tolist(),label='prediction')
    plt.plot(yhat, label='smooth')

    finaArra = []
    finaArra.append(t['y'].tolist())
    finaArra.append(t['pred'].tolist())
    import csv
    with open("3_1000deep_first_col_is_expected_second_col_is_predicted.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(finaArra)
    """
    HI PRAYAG

    t['y'].tolist() this is the expected data > BLUE
    t['pred'].tolist() this is the predicted data > Orange

    """

    plt.ylabel('output')
    plt.legend()
    plt.show()
# Plot the chart
chart_regression(pred.flatten(),y_test)

import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
