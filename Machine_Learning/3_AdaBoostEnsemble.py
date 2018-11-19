from sklearn.ensemble import AdaBoostClassifier

"""






 Broken, Ignore.








"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
# #############################################################################
# Generate some sparse data to play with
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

res = fina[51:]
test = fina[1:50]
X_train = []
y_train = []
for row in res:
    X_train.append(row[0:12])
    y_train.append(row[12])

X_test = []
y_test = []
for row in test:
    X_test.append(row[0:12])
    y_test.append(row[12])

#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)


# #############################################################################
print("="*50)
# Lasso
from sklearn.linear_model import Ridge

clf = AdaBoostClassifier

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv=5)
scores.mean()
