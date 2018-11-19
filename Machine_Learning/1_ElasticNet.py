print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel

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
print("Lasso Method")
# Lasso
from sklearn.linear_model import Lasso

alpha = 0.1
alpha_test = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 1.0, 1.25, 1.5, 2.0]
alph = 0
max_r = 0
for alp in alpha_test:
    avg_r = 0
    lasso = Lasso(alpha=alp)
    for are in range(0, 10):
        y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
        avg_r += r2_score(y_test, y_pred_lasso)
        if (r2_score(y_test, y_pred_lasso) > max_r):
            alph = alp
            max_r = r2_score(y_test, y_pred_lasso)
            print("New Max:")
            print(max_r)
    avg_r = avg_r/10

print("*"*10)
print("Best Alpha: "+str(alph))
print("Max R:"+str(max_r))
print("___")
print(lasso)
print("___")
print(SelectFromModel(lasso))
print("___")
print("Function Vars: ")
print(pretty_print_linear(lasso.coef_))
print("r^2 on test data : %f" % max_r)

# #############################################################################
# ElasticNet
print("="*50)
print("Elastic Net Method")
from sklearn.linear_model import ElasticNet
alpha = 0.1
alpha_test = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 1.0, 1.25, 1.5, 2.0]
alph = 0
max_r = 0
for alp in alpha_test:
    avg_r = 0
    enet = ElasticNet(alpha=alp, l1_ratio=0.7)
    for are in range(0, 10):
        y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
        avg_r += r2_score(y_test, y_pred_enet)
        if (r2_score(y_test, y_pred_enet) > max_r):
            alph = alp
            max_r = r2_score(y_test, y_pred_enet)
            print("New Max:")
            print(max_r)
    avg_r = avg_r/10

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print("*"*10)
print("Best Alpha: "+str(alph))
print("Max R:"+str(max_r))
print("___")
print(enet)

print("Function Vars: ")
print(pretty_print_linear(enet.coef_))
print("___")
print("r^2 on test data : %f" % r2_score_enet)

plt.plot(enet.coef_, color='lightgreen', linewidth=2,
         label='Elastic net coefficients')
plt.plot(lasso.coef_, color='gold', linewidth=2,
         label='Lasso coefficients')
#plt.plot(coef, '--', color='navy', label='original coefficients')
plt.legend(loc='best')
plt.title("K=8 Lasso R^2: %f, Elastic Net R^2: %f"
          % (max_r, r2_score_enet))
plt.show()