print(__doc__)

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
print("Ridge Regression")
# Lasso
from sklearn.linear_model import Ridge

alpha = 0.1
alpha_test = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 1.0, 1.25, 1.5, 2.0]
alph = 0
max_r = 0
for alp in alpha_test:
    avg_r = 0
    reg = Ridge(alpha=alp)
    for are in range(0, 10):
        y_pred_lasso = reg.fit(X_train, y_train).predict(X_test)
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
print(reg)
print("___")
print(SelectFromModel(reg))
print("___")
print("Function Vars: ")
print(pretty_print_linear(reg.coef_))
print("r^2 on test data : %f" % max_r)

plt.plot(reg.coef_, color='gold', linewidth=2,
         label='Ridge coefficients')
#plt.plot(coef, '--', color='navy', label='original coefficients')
plt.legend(loc='best')
plt.title("K=8 Ridge R^2: %f"
          % (max_r))
plt.show()

# #############################################################################
print("="*50)
# Lasso
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

print("SVR Kernel = 'RBF'")
y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)

count = 0
SSE = 0
for val in y_rbf:
    print(val)
    print(y_test[count])

    SSE += (y_test[count]-val)**2

    count += 1

def y_mean(arr):
    lengt = float(len(arr))
    sum = 0.0
    for val in arr:
        sum += float(val)

    return (sum/lengt)

print("mean: "+str(y_mean(y_test)))
y_meanA = y_mean(y_test)
count = 0
SSyy = 0
for val in y_rbf:
    print(val)
    print(y_test[count])

    SSyy += (y_test[count]-y_meanA)**2

    count += 1

print("*"* 50)
print("Final R_Squared: "+str((1.0 - SSE/SSyy)))

print("SVR Kernel = 'Poly'")
y_rbf = svr_poly.fit(X_train, y_train).predict(X_test)
count = 0
SSE = 0
for val in y_rbf:
    print(val)
    print(y_test[count])

    SSE += (y_test[count]-val)**2

    count += 1

def y_mean(arr):
    lengt = float(len(arr))
    sum = 0.0
    for val in arr:
        sum += float(val)

    return (sum/lengt)

print("mean: "+str(y_mean(y_test)))
y_meanA = y_mean(y_test)
count = 0
SSyy = 0
for val in y_rbf:
    print(val)
    print(y_test[count])

    SSyy += (y_test[count]-y_meanA)**2

    count += 1

print("*"* 50)
print("Final R_Squared: "+str((1.0 - SSE/SSyy)))
print("*"*10)
print("Best Alpha: "+str(alph))
print("Max R:"+str(max_r))
print("___")
print(reg)
print("___")
print(SelectFromModel(reg))
print("___")
print("Function Vars: ")
print("r^2 on test data : %f" % max_r)

plt.scatter(X_test, y_test, color='darkorange', label='data')