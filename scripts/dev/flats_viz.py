#Funkar inte!!

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

from explainerdashboard import RegressionExplainer,ClassifierExplainer, ExplainerDashboard


# ---Import data
flats = pd.read_csv('/home/chpatola/Desktop/Skola/Python/turku_flatprices/data/original/turkuflats.csv', sep=',')
flats.describe()

# ---Cleaning

# Rename columns
colindexes = np.arange(12)
newnames = ['District','Rooms','Type','M2',
                'SellPrice','€/M2','BuiltYear','Floor',
                'Elevator','State','Plot','EnergyClass']
flats.rename(columns=dict(
                        zip(flats.columns[colindexes],newnames)),inplace=True)

#Drop unnessecary columns
flats.drop(['Type','€/M2','Plot'],axis =1,inplace=True)

#Unify column contents
flats['Rooms'] = flats['Rooms'].astype(str).str[0]
flats['Floor'] = flats['Floor'].astype(str).str[0]
flats['EnergyClass'] = flats['EnergyClass'].astype(str).str[0]


flats['District']=flats.District.str.replace(r'(^.*linnanf.*$)', 'Linnanfältti')
flats['District']=flats.District.str.replace(r'(^.*eskusta.*$)', 'Keskusta')
flats['District']=flats.District.str.replace(r'(^.*aunistula.*$)', 'Raunistula')

flats.replace({'EnergyClass':{'A':'A-C','B':'A-C','C':'A-C',
                    'D':'D-G','E':'D-G','F':'D-G','G':'D-G','n':'D-G'},
                'Rooms': {'A':flats['Rooms'].mode()[0],
                    'a':flats['Rooms'].mode()[0], 
                    'h':flats['Rooms'].mode()[0]},
                'Floor': {'n':flats['Floor'].mode()[0],
                     '-':flats['Rooms'].mode()[0]}},inplace=True) 

flats[flats.District.str.contains('eskust')]

#Make values numerical
flats.replace({'State':{'bra':3,'nöjaktig':2,'dålig':1}},inplace=True) 

flats['M2'] = flats['M2'].str.replace(',','.')
flats['M2'] = flats['M2'].astype(float)

flats.Rooms =flats.Rooms.astype(int)
flats.Floor =flats.Floor.astype(int)

#Handle NA 
flats['StateNA'] = np.where(flats['State'].isna(), True, False)
flats['State'] = flats['State'].fillna(flats['State'].mode()[0])
flats.dtypes

#Transform categorical to dummies
flats_dummified = pd.get_dummies(flats)

#---Divide into x and y
X = flats_dummified.drop('SellPrice',axis=1)
y = flats_dummified.SellPrice


#---Split into train and test
train_X, test_X, train_y, test_y = train_test_split(X, 
    y,test_size = 0.3,random_state = 2)

print(train_y.all())
print(test_y.any())
print(X)
#--- Picture model
ExplainerDashboard(ClassifierExplainer(RandomForestClassifier().fit(train_X, train_y),test_X,test_y)).run()  