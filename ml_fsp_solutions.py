"""
Machine Learning with Python - FSP Assignment (Sem 4)
All 10 solutions in a single file.

Install:  pip install pandas numpy scikit-learn matplotlib seaborn
Run all:  python ml_fsp_solutions.py
Run one:  python ml_fsp_solutions.py 5      (runs only Question 5)
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import (fetch_california_housing, load_breast_cancer,
                              load_diabetes, load_iris, load_wine)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Q1: Load dataset, 70/30 train/test split, StandardScaler
# ---------------------------------------------------------------------------
def q1():
    df = load_wine(as_frame=True).frame
    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)  # no refit on test

    print("Train shape:", X_train_scaled.shape)
    print("Test shape :", X_test_scaled.shape)
    print("\nScaled train mean (~0):\n", X_train_scaled.mean().round(3).head())
    print("\nScaled train std (~1):\n", X_train_scaled.std().round(3).head())


# ---------------------------------------------------------------------------
# Q2: Boxplot + scatter plot to identify outliers and distribution patterns
# ---------------------------------------------------------------------------
def plot_outliers(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(y=df[column], ax=axes[0], color="skyblue")
    axes[0].set_title(f"Boxplot of {column}")

    sns.scatterplot(x=df.index, y=df[column], ax=axes[1], color="tomato")
    axes[1].set_title(f"Scatter plot of {column} (by row index)")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel(column)

    # Report outliers using the IQR rule
    q1_, q3_ = df[column].quantile([0.25, 0.75])
    iqr = q3_ - q1_
    low, high = q1_ - 1.5 * iqr, q3_ + 1.5 * iqr
    outliers = df[(df[column] < low) | (df[column] > high)]
    print(f"IQR bounds: [{low:.3f}, {high:.3f}] -> {len(outliers)} outliers")

    plt.tight_layout()
    plt.savefig("q02_outlier_plot.png", dpi=150)
    plt.show()


def q2():
    df = load_diabetes(as_frame=True).frame
    plot_outliers(df, "bmi")


# ---------------------------------------------------------------------------
# Q3: Linear Regression on housing data -> RMSE, MAE, R2
# ---------------------------------------------------------------------------
def q3():
    housing = fetch_california_housing(as_frame=True)  # downloads once, then cached
    X, y = housing.data, housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")


# ---------------------------------------------------------------------------
# Q4: Iris + LogisticRegression, predicted classes for the test set
# ---------------------------------------------------------------------------
def q4():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Predicted classes :", y_pred)
    print("Actual classes    :", y_test)
    print("Predicted names   :", [str(iris.target_names[i]) for i in y_pred[:10]], "...")
    print(f"Test accuracy     : {clf.score(X_test, y_test):.4f}")


# ---------------------------------------------------------------------------
# Q5: KNN (k=5) + confusion matrix
# ---------------------------------------------------------------------------
def q5():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap="Blues")
    plt.title("KNN (k=5) Confusion Matrix")
    plt.savefig("q05_confusion_matrix.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Q6: Decision Tree vs Random Forest (binary classification)
# ---------------------------------------------------------------------------
def q6():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    print(f"Decision Tree accuracy : {dt_acc:.4f}")
    print(f"Random Forest accuracy : {rf_acc:.4f}")

    if rf_acc > dt_acc:
        print("Random Forest performed better.")
    elif dt_acc > rf_acc:
        print("Decision Tree performed better.")
    else:
        print("Both models performed equally.")


# ---------------------------------------------------------------------------
# Q7: Precision, recall, F1 for a specific class (no classification_report)
# ---------------------------------------------------------------------------
def q7():
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
    target_class = 1

    # Method 1: manual calculation
    tp = sum(t == target_class and p == target_class for t, p in zip(y_true, y_pred))
    fp = sum(t != target_class and p == target_class for t, p in zip(y_true, y_pred))
    fn = sum(t == target_class and p != target_class for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"[Manual]  Class {target_class}: precision={precision:.4f}, "
          f"recall={recall:.4f}, F1={f1:.4f}")

    # Method 2: scikit-learn
    p = precision_score(y_true, y_pred, pos_label=target_class)
    r = recall_score(y_true, y_pred, pos_label=target_class)
    f = f1_score(y_true, y_pred, pos_label=target_class)
    print(f"[sklearn] Class {target_class}: precision={p:.4f}, recall={r:.4f}, F1={f:.4f}")


# ---------------------------------------------------------------------------
# Q8: 5-fold cross-validation of Ridge regression
# ---------------------------------------------------------------------------
def q8():
    X, y = load_diabetes(return_X_y=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge = Ridge(alpha=1.0)

    scores = cross_val_score(ridge, X, y, cv=kf, scoring="r2")

    print("R2 per fold :", np.round(scores, 4))
    print(f"Mean R2     : {scores.mean():.4f}")
    print(f"Std dev     : {scores.std():.4f}  (lower = more stable)")


# ---------------------------------------------------------------------------
# Q9: GridSearchCV for Random Forest
# ---------------------------------------------------------------------------
def q9():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print("Best parameters   :", grid.best_params_)
    print(f"Best CV accuracy  : {grid.best_score_:.4f}")
    print(f"Test accuracy     : {grid.best_estimator_.score(X_test, y_test):.4f}")


# ---------------------------------------------------------------------------
# Q10: Pipeline = MinMaxScaler -> LogisticRegression
# ---------------------------------------------------------------------------
def q10():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipe.fit(X_train, y_train)      # fits scaler on train, then classifier
    y_pred = pipe.predict(X_test)   # scales test with train stats, then predicts

    print(f"Pipeline test accuracy: {accuracy_score(y_test, y_pred):.4f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
QUESTIONS = {1: q1, 2: q2, 3: q3, 4: q4, 5: q5, 6: q6, 7: q7, 8: q8, 9: q9, 10: q10}

if __name__ == "__main__":
    to_run = [int(a) for a in sys.argv[1:]] or list(QUESTIONS)
    for n in to_run:
        print(f"\n{'=' * 60}\nQuestion {n}\n{'=' * 60}")
        try:
            QUESTIONS[n]()
        except Exception as e:  # one failure shouldn't stop the rest
            print(f"Question {n} failed: {e}")
