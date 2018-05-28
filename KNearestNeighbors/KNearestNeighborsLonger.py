# K Nearest Neighbors FROM SCRATCH

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warning
from matplotlib import style
from collections import Counter
import pandas as pd
import random
style.use("fivethirtyeight")

# data_set = ["k":[[1,2],[2,3],[3,1]], "r":[[6,5],[7,7],[8,6]]}
# new_features = [5,7]

# [[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1])
# plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warning.warn("K is set to a value less than the total voting groups!")
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distance.append([euclidean_distance, group])
            
    votes = [i[1] for i in sorted(distances)[:k]]
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence
    
# result = k_nearest_neighbors(dataset, new_features, k=3)
# print(result)
#
# [[plt.scatter(ii[0],ii[1],s=100, colors=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], color=result)
# plt.show()

accuracies = []
# Number of tests
for i in range(5):
    # Connecting to dataset from seperate file in same directory
    df = pd.read_csv("breast_cancer_wisconsin.data.txt")
    df.replace("?", -99999, inplace=True)
    # Removes column that would NEGATIVELY imppact result
    df.drop(["id"], 1, inplace=True)
    full_data = df.astype(float).value.to_list()
    random.shuffle(full_data)
    
    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    # 1st 20% of data
    train_data = full_data[:-int(test_size*len(full_data))]
    # Last 20% of data
    test_data = full_data[-int(test_size*len(full_data)):]
    
    # Populates our dictionaries
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
        
    correct = 0
    total = 0
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            # else:
            #    confidence of votes we got incorrect
             #   print(confidence)
            total += 1
    # print("Accuracy: ", correct/total)
    accuracies.append(correct/total)
    
print(sum(accuracies)/len(accuracies))
