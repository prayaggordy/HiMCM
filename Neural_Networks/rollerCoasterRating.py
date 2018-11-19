# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

os.environ["PATH"] += os.pathsep + 'file:///home/gautom/anaconda3/envs/pyenv35/lib/graphviz/'
import csv
zaza = []
with open("joinedTestingData.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        zaza.append(row)

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

forda = results
results = remove_col(results, 0)
results = remove_col(results, 0)

resul = results[1:]
susa = resultsTwo[1:]
for row in susa:
    resul.append(row)


fina = []
for row in resul:
    rowa = []
    for val in row:
        rowa.append(float(val))
    fina.append(rowa)

res = fina[1:]
X_train = []
y_train = []
for row in res:
    print(row)
    X_train.append(row[0:11])
    y_train.append(row[11])
import numpy as np
preds = np.asarray(X_train)
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
scores = loaded_model.predict(preds)

count = 0
final_out = []
final_out.append(["Name", "Theirs", "Ours"])
zaza = zaza[1:]
for val in scores:
    rows = []
    print("*"*10)
    print(count)
    rows.append(zaza[count][0])
    rows.append(y_train[count])
    rows.append(val.tolist()[0])
    print(rows)
    count += 1
    final_out.append(rows)

import csv
with open("finalRatings.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(final_out)
