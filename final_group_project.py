#final group project
#predict the sales_cat variable using a decision tree with a max depth of 1, min n of 2, and seed of 1234

import pandas as pd
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report

# --- Load both files ---
carseat_train = pd.read_csv('carseats_train.csv')
carseat_test = pd.read_csv('carseats_test.csv')

# --- Separate predictors (X) from target (y) ---
# Drop BOTH 'sales' (the raw number sales_cat came from) and 'sales_cat' itself
X_train = carseat_train.drop(columns=['sales', 'sales_cat'])
y_train = carseat_train['sales_cat']

X_test = carseat_test.drop(columns=['sales', 'sales_cat'])
y_test = carseat_test['sales_cat']

# --- Preprocessing: one-hot encode the 3 categorical columns ---
categorical_cols = ['shelf_location', 'urban', 'us']

column_transformer = ColumnTransformer(
    [
        ('dummify', OneHotEncoder(sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'  # leave the numeric columns untouched
)

# --- Full pipeline: preprocessing + model together ---
decision_tree_pipeline = Pipeline([
    ('preprocessing', column_transformer),
    ('dtree', DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=2,   # "min n of 2"
        random_state=1234
    ))
])

# --- Fit on TRAINING data only ---
decision_tree_pipeline.fit(X_train, y_train)

# --- Predict on TEST data ---
y_pred = decision_tree_pipeline.predict(X_test)

# --- Confusion matrix ---
# labels=['High','Low'] makes the row/column order explicit and predictable
cm = confusion_matrix(y_test, y_pred, labels=['High', 'Low'])
print("Confusion matrix (rows=actual, cols=predicted), order = [High, Low]:")
print(cm)

tp, fn, fp, tn = cm.ravel()  # careful: with labels=['High','Low'], "High" is being treated as the positive class here
print(f"\nTP={tp}  FN={fn}  FP={fp}  TN={tn}")

print("\nscikit-learn's report (to check your by-hand math):")
print(classification_report(y_test, y_pred, labels=['High', 'Low']))