from sklearn.model_selection import train_test_split

import pandas as pd

# Preprocessing
names = ['label', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor',
         'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
         'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring',
         'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

shrooms = pd.read_csv('Mushrooms.txt', skiprows=9, header=None, names=names,
                      skipfooter=1, na_values=['?'], engine='python')

shrooms = shrooms.dropna()  # Removes about a fourth of the rows for this data set

X = shrooms.iloc[:, 1:]
y = shrooms.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train and evaluate
import Oblig1.DecisionTree as tree
root = tree.learn(X_train, y_train, impurity_measure='gini', pruning=True)  # pruning happens inplace
predicted = tree.predict(X_test, root)
matching = 0
for i in range(len(predicted)):
    if predicted[i] == y_test.tolist()[i]:
        matching += 1

tree.print_tree(root, show_variables=False)
print('Accuracy on test set: ', matching/len(predicted))

# Compare to sklearn model here
print()
print('Comparison to sk-learn')
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
df = shrooms.apply(LabelEncoder().fit_transform)
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print('Accuracy on test set for DecisionTreeClassifier:', score)
