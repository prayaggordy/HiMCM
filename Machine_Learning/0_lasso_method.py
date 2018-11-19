from sklearn import linear_model

clf = linear_model.Lasso(alpha=0.1)

import csv

results = []
with open("real_final.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)

def remove_col(arra, index):
    for row in arra:
        del row[index]

    return arra
results = remove_col(results, 0)
print(results[0])
resul = results[1:]

fina = []
for row in resul:
    rowa = []
    print(row)
    for val in row:
        rowa.append(float(val))
    fina.append(rowa)


import random
random.shuffle(fina)
random.seed(128)

res = fina[51:]
test = fina[1:50]
x = []
y = []
for row in res:
    print("Train")
    print(row[12])
    print(row[0:11])
    x.append(row[0:11])
    y.append(row[12])

clf.fit(x, y)

x = []
y = []
for row in test:
    print("Test")
    print(row[12])
    print(row[0:11])
    x.append(row[0:11])
    y.append(row[12])

pred = clf.predict(x)
count = 0
SSE = 0
for val in pred:
    print(val)
    print(y[count])

    SSE += (y[count]-val)**2

    count += 1

def y_mean(arr):
    lengt = float(len(arr))
    sum = 0.0
    for val in arr:
        sum += float(val)

    return (sum/lengt)

print("mean: "+str(y_mean(y)))
y_meanA = y_mean(y)
count = 0
SSyy = 0
for val in pred:
    print(val)
    print(y[count])

    SSyy += (y[count]-y_meanA)**2

    count += 1

print("*"* 50)
print("Final R_Squared: "+str((1.0 - SSE/SSyy)))
