import pandas
from pandas import read_csv
from pandas.plotting import andrews_curves
import matplotlib.pyplot as plt

data = read_csv('real_final.csv')
plt.figure()
andrews_curves(data, 'name', colormap='gist_rainbow')
plt.show()