# carseats-sales-classification

This project builds a simple classification model to predict the `sales_cat` variable using a decision tree trained on the Carseats dataset.

## Objective

The goal of this project is to demonstrate a basic supervised learning workflow for binary classification using Python and scikit-learn.

## Project Summary

The script performs the following steps:
- loads the training and test CSV files,
- separates the feature variables from the target variable,
- converts categorical variables using one-hot encoding,
- trains a decision tree classifier with a maximum depth of 1 and a minimum of 2 samples per leaf,
- evaluates model performance using a confusion matrix and classification report.

## Files Included

- `final_group_project.py` — main Python script for training and evaluation
- `carseats_train.csv` — training dataset
- `carseats_test.csv` — test dataset

## Requirements

Make sure the following Python packages are installed:

```bash
pip install pandas scikit-learn
```

## Running the Script

From the project folder, run:

```bash
python final_group_project.py
```

If you are using the workspace virtual environment, run:

```bash
.venv\Scripts\python.exe final_group_project.py
```

## Output

When the script runs, it prints:
- the confusion matrix,
- the TP, FN, FP, and TN values,
- a classification report from scikit-learn.

## Example Results

A typical run will display a confusion matrix and a classification report for the test set. The model uses the following configuration:
- decision tree classifier
- max depth = 1
- min samples per leaf = 2
- random state = 1234

## Notes

This repository version is intended to be a clear and reproducible example of a decision tree classification workflow suitable for GitHub upload and sharing.
