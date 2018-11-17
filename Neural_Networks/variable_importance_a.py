#Random Forest
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
root_dir = os.path.abspath('./')
df = pd.DataFrame.from_csv(os.path.join(root_dir, 'finalTrainingData.csv'), sep=',', index_col=None)
print(df.columns.tolist())
print(df)
data = df[['name', 'park', 'country', 'status', 'manufacturer', 'construction', 'launch', 'restraint', 'type', 'opening_year', 'height', 'speed', 'length', 'inversions']]
data = pd.get_dummies(data)

#Selected columns based on feature selection
#selectedColumns = ['Timepoint', 'Time', 'Dose', 'Event_a', 'Event_b', 'Event_f','SurgeryType']
#data = data[selectedColumns]

print(data.columns.tolist())

# Specify the data
X = data.drop('SurgeryType', 1)
# Specify the target labels and flatten the array
y=np.array(data['score'].astype(int))
nb_classes=4


Y=y
#from keras.utils import np_utils
#Y = np_utils.to_categorical(y, nb_classes)


# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=9)

# Feature scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)

# Fitting classifier
model = RandomForestClassifier(n_estimators = 150, criterion = 'entropy')
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

print("RandomForestClassifier Accuracy: ", model.score(X_test, y_test))

from sklearn import metrics
#probs = model.predict_proba(X_test)
#fpr, tpr, threshs = metrics.roc_curve(y_test['SurgeryType'], probs[0][:,1])
#-----------------------ROC_--------------------------------
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn.metrics import roc_curve, auc

n_classes=4
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
classes=[0, 1, 2, 3]
#print(y_test)
#print(y_pred)
y_test1 = label_binarize(y_test, classes=classes)
y_pred1 = label_binarize(y_pred, classes=classes)
n_classes = len(classes)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], y_pred1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test1.ravel(), y_pred1.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print("ROC _ AUC" , roc_auc["micro"])
#--------------------------------------
# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Predicted Nicotine Exposure Age classes')
plt.legend(loc="lower right")
plt.show()
#----------------------------------------

#---------------
definitions = ['None','P33','P47','P61']
reversefactor = dict(zip(range(4),definitions))
print(reversefactor)
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
#print(y_test)
#print(y_pred)
print("#-------------------Confusion Matrix---------------------------------#")
print(pd.crosstab(y_test,y_pred, rownames=['Actual Exposure Age'], colnames=['Predicted Exposure']))
print("#----------------------------------------------------#")
# View a list of the features and their importance scores
print("List of the features and their importance scores")
print(list(zip(X_train, model.feature_importances_)))
print("#----------------------------------------------------#")

# Set the style
# Get numerical feature importances
one_importances = list(model.feature_importances_)
#plt.style.use('fivethirtyeight')
# list of x locations for plotting
one_x_values = list(range(len(one_importances)))

# Load Dataset
root_dir = os.path.abspath('./')
df = pd.DataFrame.from_csv(os.path.join(root_dir, 'second_final.csv'), sep=',', index_col=None)
df  =  df[(df["AgeStart"] == 1) ]
names = ['Event','Timepoint', 'Dose', 'SurgeryType']
print(df.columns.tolist())

data = df[names]
data = pd.get_dummies(data)

print(data.columns.tolist())

X = data.drop('SurgeryType', 1)
y=np.array(data['SurgeryType'].astype(int))
nb_classes=4


Y=y
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


model = RandomForestClassifier(n_estimators = 150, criterion = 'entropy')
model.fit(X_train, y_train)
two_importances = list(model.feature_importances_)
two_x_values = list(range(len(two_importances)))

count = 0
importance = []
print(one_importances)
print(two_importances)
for row in one_importances:
    res = one_importances[count]-two_importances[count]
    importance.append(res)
    count+=1




# Make a bar chart
plt.bar(one_x_values, one_importances, orientation = 'vertical')
# Tick labels for x axis
feature_list=X_train
plt.xticks(one_x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

plt.axhline(0, color='black')
#plt.axvline(0, color='black')
plt.show()

# Make a bar chart
plt.bar(two_x_values, two_importances, orientation = 'vertical')
# Tick labels for x axis
feature_list=X_train
plt.xticks(one_x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

plt.axhline(0, color='black')
#plt.axvline(0, color='black')
plt.show()

# Make a bar chart
plt.bar(one_x_values, importance, orientation = 'vertical')
# Tick labels for x axis
feature_list=X_train
plt.xticks(one_x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

plt.axhline(0, color='black')
#plt.axvline(0, color='black')
plt.show()

