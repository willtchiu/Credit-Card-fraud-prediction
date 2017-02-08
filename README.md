# Credit-Card-fraud-prediction
Using simple logistic regression to predict credit card fraud transactions. Data set obtained from: https://www.kaggle.com/dalpozz/creditcardfraud

Data Set: creditcard.csv classes (column 31) are strings and csvread assumes numerical data, so I preprocessed a little by changing all class identifiers (e.g. "0", "1") to numerical 0 or 1. Easy to do in vim, run:
```
:%s/,"0"/,0/gi
:%s/,"1"/,1/gi
```

## Execute: run ccDriver.m

## UPDATE: So far 84% Precision, 80% Recall, and 99.9% overall accuracy

### TODO List:
```diff
+ Created under-sample of data for data-skew classification
+ Trained logistic regression classifier with under-sample data
+ Added precision/recall analysis
+ Executed multiple iterations per value of lambda (10000 for under_sample, and 3000 for entire data set)
- Graph learning curves to detect bias/variance issue (WIP)
- Use SMOTE and RF to push precision/recall to 98%+ (WIP)
````
