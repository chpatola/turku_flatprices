import pandas as pd

import numpy as np

from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import sklearn.metrics
from sklearn.metrics import average_precision_score
import seaborn as sns
from sklearn.feature_selection import SelectFromModel

def histP_df(data):
    data.hist()
    pyplot.show()

def scattP(xdata,ydata,label):
    sns.scatterplot(x=xdata, y=ydata,hue=label)
    pyplot.show()    


# ---Import and inspect data

flats = pd.read_csv('/home/chpatola/Desktop/Skola/Python/turku_flatprices/data/original/turkuflats_2021_06.csv', sep=',')

flats.dtypes
flats.describe()

flats.columns
flats.head

for c in flats.columns:
    print(flats[c].unique())  


# ---Basic Cleaning

# Rename columns
colindexes = np.arange(12)
newnames = ['District','Rooms','Type','M2',
                'SellPrice','€/M2','BuiltYear','Floor',
                'Elevator','State','Plot','EnergyClass']
print(colindexes)
flats.rename(columns=dict(zip(flats.columns[colindexes],newnames)),inplace=True)

flats.head
flats.EnergyClass
scattP(flats.M2,flats.SellPrice,flats.EnergyClass)

#Drop unnessecary columns
flats.drop(['Type','€/M2'],axis =1,inplace=True)
#'Plot','EnergyClass' too?

#Unify column contents
flats['Rooms'] = flats['Rooms'].astype(str).str[0]
flats['Floor'] = flats['Floor'].astype(str).str[0]
flats['EnergyClass'] = flats['EnergyClass'].astype(str).str[0]


flats['District']=flats.District.str.replace(r'(^.*linnanf.*$)', 'Linnanfältti')
flats['District']=flats.District.str.replace(r'(^.*eskusta.*$)', 'Keskusta')
flats['District']=flats.District.str.replace(r'(^.*aunistula.*$)', 'Raunistula')

flats[flats.District.str.contains('eskust')]
scattP(flats.M2,flats.SellPrice,flats.Elevator)
scattP(flats.M2,flats.SellPrice,flats.Plot)

#Limit data to be included in model
flats.groupby('Rooms').size() #Include only 1-4 rooms
flats.groupby('Floor').size() # remove n floor
flats.groupby('District').size() # min 6 instances, Turku away?

flats = flats[flats.Rooms.isin(['1','2','3','4'])]
flats = flats.loc[(flats.Floor != 'n') & (flats.Floor != '-')]
flats = flats.loc[(flats.District != 'Turku')]
flats = flats.groupby('District').filter(lambda x: len(x) >5)
flats = flats.loc[(flats.SellPrice < 500000)] #Perhaps better not to cap?

flats.describe()

# --- Ny column with info on if state was not reported
flats['StateNA'] = np.where(flats['State'].isin(['\xa0','NA','n']),True,False)

#--- Unify EnergyClasses and handle na in columns
flats.replace({'EnergyClass':{'A':'A-C','B':'A-C','C':'A-C',
                    'D':'D-G','E':'D-G','F':'D-G','G':'D-G','n':'D-G'},
                'Rooms': {'t':flats['Rooms'].mode()[0],
                    'o':flats['Rooms'].mode()[0], 
                    'h':flats['Rooms'].mode()[0]},
                'Floor': {'t':flats['Floor'].mode()[0],
                     '\xa0':flats['Rooms'].mode()[0]},
                'State':{'\xa0':flats['State'].mode()[0]}
                },inplace=True) 

#Make values numerical
flats.replace({'State':{'hyvä':3,'tyyd.':2,'huono':1}},inplace=True) 

flats['M2'] = flats['M2'].str.replace(',','.')
flats['M2'] = flats['M2'].astype(float)

flats.Rooms =flats.Rooms.astype(int)
flats.Floor =flats.Floor.astype(int)

#Handle NA 
flats.isna().sum()
flats.District.mode()
flats.agg(['count', 'size', 'nunique']) 

flats['State'] = flats['State'].fillna(flats['State'].mode()[0])
flats['Plot'] = flats['Plot'].fillna(flats['Plot'].mode()[0])

#Visualize

histP_df(flats)

# Feature Engineering

flats['NewBuild'] = flats['BuiltYear'].apply(lambda x: 'True' if x > 2009 else 'False')

#flats.drop(['Floor','BuiltYear'],axis =1,inplace=True)

#Make Energy class in two groups or remove?

flats.shape
#Transform categorical to dummies

flats_dummified = pd.get_dummies(flats)
flats_dummified.head()
flats_dummified.dtypes

#---Divide into x and y
flats_dummified.columns

X = flats_dummified.drop('SellPrice',axis=1)
y = flats_dummified.SellPrice

#---Split into train and test

train_X, test_X, train_y, test_y = train_test_split(X, 
    y,test_size = 0.2,random_state = 2)

#-- Standardize with minmax
scaler = MinMaxScaler()

scaler.fit(train_X)

train_X_scaled = scaler.transform(train_X)
#train_y_scaled = scaler.transform(train_y.reset_index().reshape(-1, 1))
print(train_y)
#print(train_y_scaled)
test_X_scaled = scaler.transform(test_X)

train_X_scaled.shape

#-- Feature selection

select = SelectFromModel(RandomForestRegressor(n_estimators=60,random_state=0),
    threshold='0.2*mean')

select.fit(train_X_scaled,train_y)
train_X_features = select.transform(train_X_scaled)

train_X_features.shape

mask = select.get_support()
train_X.iloc[1,mask]


# Grid search

# Number of trees in random forest
n_estimators = [200,500,800]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

grid_search= GridSearchCV(RandomForestRegressor(),
                random_grid, cv=4, scoring
                = 'neg_median_absolute_error',return_train_score=True)
grid_search.fit(train_X,train_y)
print(grid_search.best_params_)  
print(grid_search.best_score_) 
#{'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 500}                           
  
#-- Testa modeller

models = []
models.append(('RandFor', RandomForestRegressor()))
models.append(('Tree', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('LR', LinearRegression()))
models.append(('xgb', xgb.XGBClassifier(subsample=0.2,min_samples_split = 75,
max_features = 'sqrt',n_estimators=8)))
models.append(('LDA', LinearDiscriminantAnalysis()))


# 7.2 evaluate each model in turn
sklearn.metrics.SCORERS.keys()
results = []
names = []
for name, model in models:
	kfold = model_selection.StratifiedKFold(n_splits=5, random_state=0, shuffle = True) #Cross-validation definition
	cv_results = model_selection.cross_val_score(model, train_X_features, train_y, cv=kfold, scoring='neg_median_absolute_error')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)    


#RandFor LDA both approx.11 000

# --- Save train_scaled to csv
train_X.shape
train_X_scaled.shape

train= pd.DataFrame(train_X_scaled)
train.columns = train_X.columns
train.shape
train['y'] = train_y.reset_index()['SellPrice']

train.to_csv('/home/chpatola/Desktop/Skola/Python/turku_flatprices/data/pre-processed/processedFlats.csv')








