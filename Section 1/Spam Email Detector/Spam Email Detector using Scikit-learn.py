# -----Developed By: Muhammad Umair Habib-----

# First we will download a spambase file from the internet.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset and split it into training and testing sets

data = pd.read_csv('spambase.csv')
X = data.drop('spam', axis = 1)
Y = data['spam']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the logistic regression model to classify emails as spam or not.

model = LogisticRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print(X_test)

# Evaluate the model using accuracy, confusion matrix, precision, recall and F1 score

accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)

print("Accuracy: ", accuracy)
print("Precision", precision)
print("Recall", recall)
print("F1", f1)


# Visualize the confusion matrix using Seaborns heatmap

cm = confusion_matrix(Y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()