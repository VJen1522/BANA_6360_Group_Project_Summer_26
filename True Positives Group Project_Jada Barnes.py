# compute_results.py
# Train the same decision tree pipeline and compute evaluation metrics

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
import matplotlib.pyplot as plt


def specificity_score(y_true, y_pred, pos_label='High'):
    # Compute specificity = TN / (TN + FP)
    # Determine labels order so we can extract TN and FP reliably
    labels = [pos_label]
    cm = confusion_matrix(y_true, y_pred, labels=[pos_label, 'Low'])
    # cm layout with labels=[pos, neg]: [[TP, FN], [FP, TN]]
    tp, fn, fp, tn = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def main():
    # Load data
    train = pd.read_csv('carseats_train.csv')
    test = pd.read_csv('carseats_test.csv')

    X_train = train.drop(columns=['sales', 'sales_cat'])
    y_train = train['sales_cat']
    X_test = test.drop(columns=['sales', 'sales_cat'])
    y_test = test['sales_cat']

    # Preprocessing
    categorical_cols = ['shelf_location', 'urban', 'us']
    column_transformer = ColumnTransformer(
        [('dummify', OneHotEncoder(sparse_output=False), categorical_cols)],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessing', column_transformer),
        ('dtree', DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, random_state=1234))
    ])

    # Fit and predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['High', 'Low'])
    print('Confusion matrix (rows=actual, cols=predicted), order = [High, Low]:')
    print(cm)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='High')
    recall = recall_score(y_test, y_pred, pos_label='High')
    spec = specificity_score(y_test, y_pred, pos_label='High')

    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Precision (High as positive): {precision:.2f}")
    print(f"Recall (High as positive): {recall:.2f}")
    print(f"Specificity (Low correctly identified): {spec:.2f}\n")

    print("scikit-learn classification report:\n")
    print(classification_report(y_test, y_pred, labels=['High', 'Low']))

    # --- Plots: confusion matrix and metrics bar chart ---
    cm_df = pd.DataFrame(cm, index=['Actual High', 'Actual Low'], columns=['Predicted High', 'Predicted Low'])
    plt.figure(figsize=(6, 4))
    plt.imshow(cm_df, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = [0, 1]
    plt.xticks(ticks, cm_df.columns)
    plt.yticks(ticks, cm_df.index)
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            plt.text(j, i, cm_df.iat[i, j], ha='center', va='center', color='black')
    plt.tight_layout()
    cm_path = 'confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()

    # Metrics bar chart
    metrics = {
        'Accuracy': accuracy,
        'Precision (High)': precision,
        'Recall (High)': recall,
        'Specificity': spec,
    }
    plt.figure(figsize=(6, 4))
    names = list(metrics.keys())
    values = list(metrics.values())
    bars = plt.bar(range(len(names)), values, color=['#4c72b0', '#55a868', '#c44e52', '#8172b2'])
    plt.ylim(0, 1)
    plt.xticks(range(len(names)), names, rotation=30)
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    metrics_path = 'metrics_bar.png'
    plt.savefig(metrics_path)
    plt.close()

    print(f"Saved confusion matrix plot to: {cm_path}")
    print(f"Saved metrics bar chart to: {metrics_path}")


if __name__ == '__main__':
    main()
