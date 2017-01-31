# Credit-Card-fraud-prediction
Using simple logistic regression to predict credit card fraud transactions. Data set obtained from: https://www.kaggle.com/dalpozz/creditcardfraud

Data Set: creditcard.csv classes (column 31) are strings and csvread assumes numerical data, so I preprocessed a little by changing all class identifiers (e.g. "0", "1") to numerical 0 or 1. Easy to do in vim, run 
```
:%s/,"0"/,0/gi
:%s/,"1"/,1/gi
```
- Created under-sample of data for data-skew classification
- Training logistic regression classifier with under-sample data (WIP)
