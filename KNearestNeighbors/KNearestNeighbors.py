# How one would use KNearest Neighbors

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd 

df = pd.read_csv("breast_cancer_wisconsin.data.txt")
df.replace("?", -99999, inplace=True)
df.drop(["id"], 1, inplace=True)

# Drops useless data that would have a NEGATIVE impact on the result
x = np.array(df.drop(["class"],1))
y = np.array(df["class"])

# Trains dataset
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)

# Sets classifier
clf = neighbors.KNeighborsClassifier(n_jobs=1)
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(len(example_measures), -1)

# If 2 then benign, if 4 malignant
prediction = clf.predict(example_measures)
print(prediction)
