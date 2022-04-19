#import necessary library
%matplotlib inline
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
import pandas as pd
import numpy as np
import patsy
from statsmodels.graphics.correlation import plot_corr
from sklearn.model_selection import train_test_split
plt.style.use ('seaborn')

#read csv file or dataset files from github
rawbostondata = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter02/Dataset/Boston.csv')

#.head show the 5 first line of our file
rawbostondata.head()

#check for null values and drop them
rawbostondata = rawbostondata.dropna()
#check for duplicate values and drop them
rawbostondata = rawbostondata.drop_duplicates()

#make a list of the headers
list(rawbostondata.columns)

#rename the headers with meaningful :) character by macking dictionary of headers
rawbostondata_rename = rawbostondata.rename (columns = {
                                                        'CRIM': '1',
                                                        ' ZN ': '2',
                                                        'INDUS ': '3',
                                                        'CHAS': '4',
                                                        'NOX': '5',
                                                        'RM': '6',
                                                        'AGE': '7',
                                                        'DIS': '8',
                                                        'RAD': '9',
                                                        'TAX': '10',
                                                        'PTRATIO': '11',
                                                        'LSTAT': '12',
                                                        'MEDV': '13' })
rawbostondata_rename.head()

#using .info help us to understand the type, count of non null, name of column of our data
rawbostondata_rename.info()

#.describe make a table of mean, std, min, 25%, 50%, 75%, max of our data
rawbostondata_rename.describe(include=[np.number]).T

# in this part of code, we train the model and then test it.
X = rawbostondata_rename.drop('1', axis=1)
y = rawbostondata_rename [['1']]
seed = 10
test_data_size = 0.3
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = test_data_size, random_state= seed)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
train_data.corr (method='pearson')

#here we make a corralation matrix and show them with a plot
corrMatrix = train_data.corr (method='pearson')

xnames = list(train_data.columns)
ynames = list(train_data.columns)
plot_corr(corrMatrix, xnames=xnames, ynames=ynames, title=None, normcolor=False, cmap='RdYlBu_r')

# in this part, we show linear relationship of our data
fig, ax = plt.subplots (figsize= (10,6))
sns.regplot (x= '13', y= '1', ci=None, data= train_data, ax= ax, color= 'k', scatter_kws= {'s':20, 'color':'royalblue', 'alpha':1})
ax.set_xlabel ('Crime rate per Capita', fontsize= 15, fontname= 'DejaVu Sans')
ax.set_ylabel ("Median value of owner-occupied homes in $1000's", fontsize= 15, fontname= 'DejaVu Sans')
ax.set_xlim (left= None, right= None)
ax.set_ylim (bottom= None, top= 30)
ax.tick_params(axis= 'both', which= 'major', labelsize= 12)
fig.tight_layout()

# in this part, we show log-linear relationship of our data with 95% confidence interval
# we use seaborn (regplot) library to creat a log-linear plot and fit a regression line through it
fig, ax = plt.subplots (figsize= (10,6))
y = np.log (train_data ['1'])
sns.regplot (x = '13', y= y, ci= 95, data= train_data, ax= ax, color= 'k', scatter_kws= {'s': 20, 'color': 'royalblue', 'alpha': 1})
ax.set_ylabel ('log of crime rate per capita', fontsize= 15, fontname= 'DejaVu Sans')
ax.set_xlabel ("Median value of owner-occupied homes in $1000's", fontsize= 15, fontname= 'DejaVu Sans')
ax.set_xlim (left= None, right= None)
ax.set_ylim (bottom= None, top= None)
ax.tick_params (axis= 'both', which= 'major', labelsize= 12)
fig.tight_layout()

