import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
        accuracy_score, 
        precision_score,
        recall_score,
        confusion_matrix,
        ConfusionMatrixDisplay
)

#1 Load the train and test data files that were provided
train_df = pd.read_csv(r'C:\Users\mta60\OneDrive\Documents\UD\2026 Summer\BANA-6360 Programming I\Project\carseats_train.csv')
test_df = pd.read_csv(r'C:\Users\mta60\OneDrive\Documents\UD\2026 Summer\BANA-6360 Programming I\Project\carseats_test.csv')

#2 Separate features and target variable from the training data
target_variable = 'sales_cat'

#2a. Use sales_cat as target answer y
y_train = train_df[target_variable]
y_test = test_df[target_variable]

#2b. Convert 'High' and 'Low'
quality_map = {'High': 1, 'Low': 0}

y_train = y_train.map(quality_map)
y_test  = y_test.map(quality_map)    

#2c. Drop both sales and sales_cat from clues
X_train = train_df.drop(columns=['sales_cat', 'sales'])
X_test = test_df.drop(columns=['sales_cat', 'sales'])

#2d. Now run get_dummies on X_train and X_test to convert remaining text columns
X_train = pd.get_dummies(X_train, drop_first=True)
X_test  = pd.get_dummies(X_test, drop_first=True)

#2e.Align columns in both test and train files
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}\n")

#3 Training
model = DecisionTreeClassifier(
    max_depth=1,
    min_samples_split=2,
    random_state=1234
)
#Learn only from training file
model.fit(X_train, y_train)
print("---Tree Training Complete---")

#4 Grade on test file
#Predict answers for the test file
predictions = model.predict(X_test)

#Model's predictions on the test set
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
specificity = tn / (tn + fp)

print(f"Test Accuracy Score: {accuracy * 100:.1f}%\n")
print(f"Precision: {precision * 100:.1f}%")
print(f"Recall: {recall * 100:.1f}%")
print(f"Specificity: {specificity * 100:.1f}%")

#Generate the raw confusion matrix array
cm = confusion_matrix(y_test, predictions)

#Plot a clean visual diagram
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot(cmap='Blues')  # 'Blues' gives a clean blue shading

# Show the chart
plt.title("Decision Tree Confusion Matrix")
plt.show()