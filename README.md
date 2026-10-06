# Machine Learning with Python – FSP Assignment

**Name:** Vivek Kumar Shaw
**Roll No.:** 13005324027
**Course:** B.Tech, Electronics and Instrumentation Engineering (5th Semester)


## About
Solutions to the 10 assignment questions on machine learning with Python, using pandas, scikit-learn, matplotlib and seaborn. All code is in a single file, `ml_fsp_solutions.py`, with one function per question.

## Questions Covered
| Q | Topic |
|---|-------|
| 1 | 70/30 train-test split and StandardScaler |
| 2 | Boxplot and scatter plot for outlier detection |
| 3 | Linear Regression on housing data (RMSE, MAE, R²) |
| 4 | Logistic Regression on the Iris dataset |
| 5 | KNN (k=5) classifier and confusion matrix |
| 6 | Decision Tree vs Random Forest |
| 7 | Precision, recall and F1-score for a specific class |
| 8 | 5-fold cross-validation with Ridge regression |
| 9 | GridSearchCV for Random Forest hyperparameters |
| 10 | Pipeline with MinMaxScaler and Logistic Regression |

## Requirements
```
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run
Run all questions:
```
python ml_fsp_solutions.py
```
Run a single question (for example Q5):
```
python ml_fsp_solutions.py 5
```

## Notes
- Q3 downloads the California housing dataset on first run, so it needs an internet connection.
- Q2 and Q5 save their plots as `.png` files in the working directory.
- All other questions use datasets built into scikit-learn.
