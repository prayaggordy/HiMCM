from sklearn import linear_model

clf = linear_model.Lasso(alpha=0.1)

import csv

results = []
with open("real_final.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)
results = [[float(numb) for numb in row] for rows in results]
res = results[51:]
test = results[1:50]
import random
random.shuffle(res)
random.seed(128)

x = []
y = []
for row in res:
    x.append(row[0:12])
    y.append(row[13])

clf.fit(x, y)

x = []
y = []
for row in test:
    x.append(row[0:12])
    y.append(row[13])

pred = clf.predict(x)
for val in pred:
    print(val)