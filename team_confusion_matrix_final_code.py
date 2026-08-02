# team_confusion_matrix_final.py
# Team True Positives — combined implementation
# Predicts sales_cat using a Decision Tree (max_depth=1, min_n=2, seed=1234)
# per the p.400 activity specification.
#
# Combines: Veronica's pipeline structure, Jada's specificity_score()
# helper and metrics chart, and Michael's ConfusionMatrixDisplay visual.

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt


def specificity_score(y_true, y_pred, pos_label='High', neg_label='Low'):
    """Specificity = TN / (TN + FP), computed directly from the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[pos_label, neg_label])
    tp, fn, fp, tn = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def main():
    # --- Load data (pre-split ~80/20 by the instructor) ---
    train = pd.read_csv('carseats_train.csv')
    test = pd.read_csv('carseats_test.csv')

    # --- Separate predictors (X) from target (y) ---
    # Drop BOTH 'sales' and 'sales_cat' from X: sales_cat is directly derived
    # from sales, so leaving sales in would let the model "see the answer."
    X_train = train.drop(columns=['sales', 'sales_cat'])
    y_train = train['sales_cat']
    X_test = test.drop(columns=['sales', 'sales_cat'])
    y_test = test['sales_cat']

    # --- Preprocessing: one-hot encode the 3 categorical columns ---
    categorical_cols = ['shelf_location', 'urban', 'us']
    column_transformer = ColumnTransformer(
        [('dummify', OneHotEncoder(sparse_output=False), categorical_cols)],
        remainder='passthrough'
    )

    # --- Model: Decision Tree per the assignment spec ---
    # max_depth=1          -> a single split ("decision stump")
    # min_samples_leaf=2   -> "min n of 2"
    # random_state=1234    -> reproducible seed
    pipeline = Pipeline([
        ('preprocessing', column_transformer),
        ('dtree', DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, random_state=1234))
    ])

    # --- Fit on TRAINING data only, predict on TEST data ---
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # --- Confusion matrix (explicit label order avoids ambiguity) ---
    cm = confusion_matrix(y_test, y_pred, labels=['High', 'Low'])
    print("Confusion matrix (rows=actual, cols=predicted), order = [High, Low]:")
    print(cm)

    tp, fn, fp, tn = cm.ravel()
    print(f"\nTP={tp}  FN={fn}  FP={fp}  TN={tn}")

    # --- Metrics ---
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / cm.sum()
    specificity = specificity_score(y_test, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    print(f"Recall:      {recall:.4f}")
    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")

    print("\nscikit-learn's report (to check our by-hand math):")
    print(classification_report(y_test, y_pred, labels=['High', 'Low']))

    # --- Visual 1: confusion matrix display (Michael's approach) ---
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['High', 'Low'])
    disp.plot(cmap='Blues')
    plt.title('Decision Tree Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # --- Visual 2: metrics bar chart (Jada's approach) ---
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'Specificity': specificity}
    plt.figure(figsize=(6, 4))
    names = list(metrics.keys())
    values = list(metrics.values())
    plt.bar(names, values, color=['#4c72b0', '#55a868', '#c44e52', '#8172b2'])
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig('metrics_bar.png')
    plt.close()

    print("\nSaved confusion_matrix.png and metrics_bar.png")


if __name__ == '__main__':
    main()
