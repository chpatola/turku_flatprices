# Turku Flatprices

## About
Predict flatprices in Turku.

Data of sold flats are taken from https://asuntojen.hintatiedot.fi/haku/. Here, flats sold within the last 12 months are listed.
The retrival was made 18.6.2021.

A RandomForestRegressor was used as predictor and the R2 score on the test data amounts to ~ 88%. This translates to the proportion of the sales price variance, that's explained by the independent values (m2, location, floor...).


## How to use
Clone this repository to your own computer.

cd to the repository.

Create a new conda environment where is the name you want to give the new environment. Then activate the enivronment

```
conda create --name <env> --file requirement.txt
  
conda activate <env> 
``` 

Set your base path at line 14 in flats.py

Run the code from the file flats.py.

```
python flats.pyy
```

The programme will print some output of the analysis. 
You can use the joblib version of the model in case you want to use it for predictions coming from an UI.