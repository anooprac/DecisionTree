import pandas as pd
from Node import *
import numpy as np
from Decision_Tree import *

data = {
    'a': np.random.randint(1, 3, 10),
    'b': np.random.randint(1, 3, 10),
    'c': np.random.randint(1, 3, 10),
    'd': np.random.randint(1, 3, 10),
    'e': np.random.randint(1, 3, 10),
    'f': np.random.randint(1, 3, 10),
    'label': np.random.randint(1, 3, 10)
}

feature_dict = dict()
feature_dict['a'] = "D"
feature_dict['b'] = "D"
feature_dict['c'] = "D"
feature_dict['d'] = "D"
feature_dict['e'] = "D"
feature_dict['f'] = "D"
df = pd.DataFrame(data)

tree = Decision_Tree_Regressor(4, 0, 1, df, feature_dict)
tree.construct_tree()
sum = 0
for i in range(0, 100):
    test_dict = dict()
    test_dict['1'] = np.random.randint(1, 11, 1)
    test_dict['2'] = np.random.randint(1, 11, 1)
    test_dict['3'] = np.random.randint(1, 11, 1)
    test_dict['4'] = np.random.randint(1, 11, 1)
    test_dict['5'] = np.random.randint(1, 11, 1)
    test_dict['6'] = np.random.randint(1, 11, 1)
    answer = np.random.randint(1, 101, size=None)
    print(tree.predict(test_dict))

