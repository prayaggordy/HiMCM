#Random Forest
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
root_dir = os.path.abspath('./')
df = pd.read_csv(os.path.join(root_dir, 'real_final.csv'), sep=',', index_col=None)
print(df.columns.tolist())
print(df)


#Selected columns based on feature selection
#selectedColumns = ['Timepoint', 'Time', 'Dose', 'Event_a', 'Event_b', 'Event_f','SurgeryType']
#data = data[selectedColumns]


print(df.columns.tolist())
# Specify the data
X = df.drop('score', axis=1)
# Specify the target labels and flatten the array
y=np.array(df['score'].astype(float))
nb_classes=4
print(X)
print(y)

Y=y
#from keras.utils import np_utils
#Y = np_utils.to_categorical(y, nb_classes)


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

