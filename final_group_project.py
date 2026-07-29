#final group project
#predict the sales_cat variable using a decision tree with a max depth of 1, min n of 2, and seed of 1234

import pandas as pd
import sklearn.pipeline
import sklearn.tree
import sklearn.compose
import sklearn.preprocessing
import sklearn.metrics

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

column_transformer = sklearn.compose.ColumnTransformer(
    [
        ('dummify', sklearn.preprocessing.OneHotEncoder(sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'  # leave the numeric columns untouched
)

# --- Full pipeline: preprocessing + model together ---
decision_tree_pipeline = sklearn.pipeline.Pipeline([
    ('preprocessing', column_transformer),
    ('dtree', sklearn.tree.DecisionTreeClassifier(
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
cm = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=['High', 'Low'])
print("Confusion matrix (rows=actual, cols=predicted), order = [High, Low]:")
print(cm)

tp, fn, fp, tn = cm.ravel()  # with labels=['High','Low'], "High" is the positive class here
print(f"\nTP={tp}  FN={fn}  FP={fp}  TN={tn}")

print("\nscikit-learn's report (to check your by-hand math):")
print(sklearn.metrics.classification_report(y_test, y_pred, labels=['High', 'Low']))