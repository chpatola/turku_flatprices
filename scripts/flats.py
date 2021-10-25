"""ML algoritm to predict flat prices in Turku, Finland"""
from os import path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

# You need to decide how you handle NA for all columns (here na is \xa0')!!

# ---Import data---

base_path = '/home/chpatola/Desktop/Skola/Python/turku_flatprices/'
input_path = path.join(base_path, 'data/original/turkuflats_2021_06.csv')
flats = pd.read_csv(input_path,
                    sep=',')
# Unique values for all columns
for col in flats:
    print(flats[col].unique())

# ---Cleaning---

# Rename columns
colindexes = np.arange(12)
newnames = ['District', 'Rooms', 'Type', 'M2',
            'SellPrice', '€/M2', 'BuiltYear', 'Floor',
            'Elevator', 'State', 'Plot', 'EnergyClass']
flats.rename(columns=dict(
    zip(flats.columns[colindexes], newnames)),
             inplace=True)

# Drop unnessecary columns
flats.drop(['Type', '€/M2', 'Plot'],
           axis=1,
           inplace=True)

# --Unify column contents--

# Take first letter in these columns
flats['Rooms'] = flats['Rooms'].astype(str).str[0]
flats['Floor'] = flats['Floor'].astype(str).str[0]
flats['EnergyClass'] = flats['EnergyClass'].astype(str).str[0]

# Unify district groups
flats['District'] = flats.District.str.replace(
    r'(^.*linnanf.*$)', 'Linnanfältti')
flats['District'] = flats.District.str.replace(r'(^.*eskusta.*$)', 'Keskusta')
flats['District'] = flats.District.str.replace(
    r'(^.*aunistula.*$)', 'Raunistula')

# New column with info on if state was not reported
flats['StateNA'] = np.where(
    flats['State'].isin(['\xa0', 'NA', 'n']), True, False)

#Unify EnergyClasses and handle na in columns
flats.replace({'EnergyClass': {'A': 'A-C', 'B': 'A-C', 'C': 'A-C',
                               'D': 'D-G', 'E': 'D-G', 'F': 'D-G', 'G': 'D-G', 'n': 'D-G'},
               'Rooms': {'t': flats['Rooms'].mode()[0],
                         'o': flats['Rooms'].mode()[0],
                         'h': flats['Rooms'].mode()[0]},
               'Floor': {'t': flats['Floor'].mode()[0],
                         '\xa0': flats['Rooms'].mode()[0]},
               'State': {'\xa0': flats['State'].mode()[0]}
               }, inplace=True)

# Make values numerical
flats.replace({'State': {'hyvä': 3, 'tyyd.': 2, 'huono': 1}},
              inplace=True)

flats['M2'] = flats['M2'].str.replace(',', '.')
flats['M2'] = flats['M2'].astype(float)

flats.Rooms = flats.Rooms.astype(int)
flats.Floor = flats.Floor.astype(int)

# ---Transform categorical to dummies---

flats_dummified = pd.get_dummies(flats)

# ---Divide into x and y---

X = flats_dummified.drop('SellPrice', axis=1)
y = flats_dummified.SellPrice


# ---Split into train and test---

train_X, test_X, train_y, test_y = train_test_split(X,
                                                    y, test_size=0.3, random_state=2)

# ---Feature selection---
select = SelectFromModel(
    RandomForestRegressor(min_samples_leaf=1,
                          min_samples_split=5, n_estimators=500,
                          random_state=0), threshold='0.2*mean')

select.fit(train_X, train_y)
train_X_features = select.transform(train_X)
test_X_features = select.transform(test_X)

mask = select.get_support()  # Chosen columns
train_X.iloc[1, mask]


# ---Test Model---

Regressor = RandomForestRegressor(min_samples_leaf=1, min_samples_split=5,
                                  n_estimators=500, random_state=0)
model = Regressor.fit(train_X_features, train_y)
score = model.score(test_X_features, test_y)
print("R2 score of the model {:2f}".format(score))

# ---Save model---
joblib.dump(model, path.join(base_path,'model_jlib'))
